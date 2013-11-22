(ns nanogenmo.core
  (:require [clojure.pprint]
            [opennlp.nlp]
            [opennlp.treebank]
            [opennlp.tools.filters]
            [opennlp.tools.train])
  (:use [clojure.pprint]
        [opennlp.nlp]
        [opennlp.treebank]
        [opennlp.tools.filters])
  (:gen-class))

(use 'clojure.java.io)

(pprint "Running...")

(def get-sentences (make-sentence-detector "models/en-sent.bin"))
(def tokenize (make-tokenizer "models/en-gutenberg-base-token.bin"))
(def detokenize (make-detokenizer "models/english-detokenizer.xml"))
(def pos-tag (make-pos-tagger "models/en-pos-maxent.bin"))
(def name-find (make-name-finder "models/namefind/en-ner-person.bin"))
(def chunker (make-treebank-chunker "models/en-chunker.bin"))
(def parser (make-treebank-parser "models/en-parser-chunking.bin"))

;;; Workaround until the bug gets fixed...
;(defmacro fixed-chunk-filter
;  "Declare a filter for treebank-chunked lists with the given name and regex."
;  [n r]
;  (let [docstring (str "Given a list of treebank-chunked elements, "
;                       "return only the " n " in a list.")]
;    `(defn ~n
;       ~docstring
;       [elements#]
;       (filter (fn [t#] (re-find ~r (:tag t#))) 
;               (remove #(nil? (:tag %)) elements#)))))

(defmacro chunk-filter-2
  "Declare a filter for treebank-chunked lists with the given name and regex."
  [n r]
  (let [docstring (str "Given a list of treebank-chunked elements, "
                       "return only the " n " in a list.")]
    `(defn ~n
       ~docstring
       [elements#]
       (filter (fn [t#] (if (nil? ~r)
                          (nil? (:tag t#))
                          (and (:tag t#)
                               (re-find ~r (:tag t#)))))
               elements#))))

(chunk-filter-2 fixed-noun-phrases #"^NP$")
;;;

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (println "Hello, World!"))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Input
;;;
;;; Bring the text in from a file and put it into a useful format.
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def ^:dynamic *windows-linebreaks* true)

(defn strip-italics [text]
  "Removes _underscored italics_ from the text. It'd be nice to find a way to include these, later."
  (clojure.string/replace text "_" ""))

(defn truncate-whitespace [text]
  "Removes excess spaces from the text, regularizing lines. Not what you want when you're trying to preserve the format of poetry, but definitely what you want when you're trying to parse the words."
  (clojure.string/replace text #"\s+" " "))

(defn strip-linebreaks-windows [text]
  "Returns string with linebreaks stripped out."
   (clojure.string/replace 
     (clojure.string/replace text #"[\r\n]+" "_@_")
     "_@_" " "))

(defn strip-linebreaks-unix [text]
  "Returns string with linebreaks stripped out."
   (clojure.string/replace 
     (clojure.string/replace text #"[\n]+" "_@_")
     "_@_" " "))

(defn strip-linebreaks [text]
  "Returns string with linebreaks stripped out."
  (if *windows-linebreaks*
    (strip-linebreaks-windows text)
    (strip-linebreaks-unix text)))

(defn mark-paragraphs-windows [source-text]
  "Find the linebreaks and mark their position for later splitting. May need updating for non-Windows files."
  (clojure.string/replace source-text #"\r\n\r\n" "¶"))

(defn mark-paragraphs-unix [source-text]
  "Find the linebreaks and mark their position for later splitting. May need updating for non-Windows files."
  (clojure.string/replace source-text #"\n\n" "¶"))

(defn mark-paragraphs [text]
  "Find the linebreaks and mark their position for later splitting."
  (if *windows-linebreaks*
    (mark-paragraphs-windows text)
    (mark-paragraphs-unix text)))

(defn break-on-pilcrow-windows [source-text]
  "Find paragraph markers and reinsert the linebreaks."
  (clojure.string/replace 
      source-text 
      #"¶" "\r\n\r\n"))

(defn break-on-pilcrow-unix [source-text]
  "Find paragraph markers and reinsert the linebreaks."
  (clojure.string/replace 
      source-text 
      #"¶" "\n\n"))

(defn break-on-pilcrow [text]
  "Find paragraph markers and reinsert the linebreaks."
  (if *windows-linebreaks*
    (break-on-pilcrow-windows text)
    (break-on-pilcrow-unix text)))

(defn append-space [source-text]
  "Add a space to the end of the string"
  (if-not (nil? (re-find #"¶" source-text))
    (str source-text " ")
    source-text))

(defn get-paragraphs [source-text]
  "Returns source-texts broken into a list of paragraphs."
  (clojure.string/split source-text #"¶"))

(defn categorize-paragraph [source-text]
  "Returns a category based on the contents of the source-text: dialog, action, or exposition."
  (cond
    (re-find #"\"" source-text) :dialogue
    (re-find #",$" source-text) :ends-with-comma ;only last sentence is dialogue
    :else :action))

(defn input-source-text-directly [source-text]
  "Takes cleaned source text and formats it into useful paragraphs."
  (get-paragraphs (truncate-whitespace
                    (strip-italics 
                      (strip-linebreaks 
                        (mark-paragraphs 
                          source-text))))))

(defn input-source-text-from-file [filename]
  "Takes cleaned source text and formats it into useful paragraphs."
  (input-source-text-directly 
    (slurp filename)))

(defn categorize-text [paragraphs]
  "Takes a vector of paragraphs and returns a categorized map of paragraphs."
  (map #(hash-map :category (categorize-paragraph %) :text %)
       paragraphs))

(defn paragraph-to-sentences [paragraph]
  "Takes paragraph-map-data and assocs the sentence breakdown."
  (assoc paragraph :sentences (get-sentences (:text paragraph))))

;; A better way to handle the dialog detection: go through the sentences, tracking the quotes 
;; as they open and close the quotations. "This is dialog," he said. This is action. "Even 
;; though they are in the same paragraph."
;; In the long run, paragraph-level analysis may be less useful than sentence-level filtering.

(defn paragraph-to-typed-sentences [paragraphs]
  "Takes a paragraph's text and adds categorized sentence breakdowns, to handle the paragraphs that mix action and dialog."
  (map #(assoc % :categorized 
               (let [sentences (get-sentences (:text %))
                     category (:category %)]
                 (cond 
                   (= category :dialogue) {:dialogue sentences}
                   (= category :action) {:action sentences}
                   (= category :ends-with-comma) {:action (butlast sentences) 
                                                  :dialogue (last sentences)})))
         paragraphs))

(defn sentences-from-paragraphs [paragraphs]
  "Given a collection of paragraphs, grab just the sentences."
  (mapcat #(:sentences %) paragraphs))

;(defn grab-sentences-of-type [paragraphs sentence-type]
;  (sentences-from-paragraphs
;         (map paragraph-to-sentences 
;              (filter #(= (:category %) sentence-type)
;                      paragraphs))))

(defn just-categorized-sentences [paragraph category]
  (category (:categorized paragraph)))

(defn append-pilcrow [sentences]
  "Given a collection of sentences, add a marker to the end of the last one."
  (if (not (nil? sentences))
    (if (> (count sentences) 0)
      (vec (conj sentences (str "¶")))
      sentences)
    sentences))

(defn get-sentences-of-category [paragraphs category]
  (mapcat
    ;#(append-pilcrow 
     #(just-categorized-sentences % category);)
    paragraphs))
  
  ;(mapcat #(category (:categorized %)) paragraphs))

  
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Actions
;;;
;;; An action is a function based on a single sentence (or other complete lexical module) 
;;; pulled from the source file.
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defstruct action :text :tokens)

(defn get-actions [source-text]
  "Take a source text, convert it to paragraphs, and finally output individual sentences as actions."
  (map #(struct action %)
       (get-sentences-of-category
         (paragraph-to-typed-sentences
           (categorize-text 
             (input-source-text-directly
               source-text)))
         :action)))

(defn just-get-sentences [source-text]
  (map #(clojure.string/replace % "`" "'")
       (input-source-text-directly
         source-text)))

(defn get-sentences-from-text [source-text]
  (get-sentences-of-category
         (paragraph-to-typed-sentences
           (categorize-text 
             (input-source-text-directly
               source-text)))
         :action))

(defn tokenize-action [act]
  (assoc act :tokens (tokenize (:text act))))

(defn dechunk [chunked-sentences unchunked-sentences]
  "Return to the unchunked sentence, with the subsitutions from the chunked sentence."
  (mapcat #(:phrase %) chunked-sentences))

(defn process-action [sentence]
  (let [tokenized-sentence (tokenize sentence)
        chunked-sentence  (chunker (pos-tag tokenized-sentence))
        unchunked-sentence tokenized-sentence]
    ;(str (detokenize unchunked-sentence) ".") ; hack to restore periods...
    (detokenize unchunked-sentence)
    ))

 
;; testing...
(comment
;(spit "texts\\output\\test.txt" 
(pprint
  (map tokenize-action
      (get-actions 
        (slurp "texts\\cleaned\\pg42671_clean.txt"))))
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Training
;;;
;;; Parsing files for training and then running the training tools on them.
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Training types:
;; Sentences: Output one sentence per line.
;; Tokenizer: One sentence per line, non-whitespace splits marked with <SPLIT>
;; Parts-of-speech: requires every part of speech be tagged, on sentence per line
;; Treebank chunker: http://www.cnts.ua.ac.be/conll2000/chunking/
;; Named entity: one tokenized sentence per line, names marked.
;; Document categorizer: Sentiment word

(defn brute-force-is-name? [token]
  "Returns true if the token is a name."
  (cond
    (re-find #"^(Countess|Lady|Mrs.|Mrs|Miss)$" token) true
    (re-find #"^(King|Queen|Duke|Duchess|Admiral|Captain|Colonel|Viscountess|Viscount|Hon.|Lord|Sir|Mr.|Mr|St.|Nurse|Doctor|Dr.|Dr|Dowager)$" token) true
    (re-find #"^(Bennets|Lucases|Harvilles|Gardiners|Collinses|Ibbotsons|Durands|Musgroves)$" token) true
    (re-find #"^(Elizabeth|Catherine|Charlotte|Caroline|Christopher|Edmund|Edward|Georgina|Henrietta|James|Lousia|Penelope|Fredrick|Fredric|William|George|Harry|Henry|Charles|Basil|Kitty|Eleanor|Elinor|Walter|Lizzy|Lydia|Louisa|Alicia|Maria|Jemima|Sarah|Jane|John|Anne|Harriet|Mary|Pen|Anna|Dick)$" token) true
    (re-find #"^(Musgrove|Wentworth|Harville|Hayter|Trent|Brigden|Musgrove|Benwick|Bennet|Bingley|Lucas|Carteret|Hurst|Smith|Darcy|Gardiner|Wickham|Collins|Rooke|Wallis|Fitzwilliam|de|Bourgh|Elliot|Russell|Shepherd|Clay|Long|Frankland|Morley|Metcalfe|Nicholls|Phillips|Pope|Pratt|Reynolds|Robinson|Stone|Watson|Webbs|Younge|Jenkinson|Jones|King|Long|Goulding|Grierson|Grantley|Haggerston|Harrington|Annesley|Chamberlayne|Dawson|Denny|Forster|Croft|Brand|Atkinson|Mackenzie|Hamilton|Dalrymple|Speed|Maclean|Ives|Spicer|Shirley|Morris)$" token) true
    :else false))
;her Ladyship, the Archbishop, the Secretary, the Superior, Mother Superior, 
;Places: Rosings, London, Pemberley, Hertfordshire, Gracechurch-street, Brighton, Meryton, Derbyshire, Rosing's Park, Lambton, Hunsford Parsonage, Netherfield, 
;Punctuation: !-- -- ;-- .--

(defn is-name? [token]
  ;(not (empty? (name-find [token]))))
  (or (not (empty? (name-find [token]))) (brute-force-is-name? token)))

(defn mark-token [predicate? tokens]
  (mapcat #(if (predicate? %)
             ["<START>" % "<END>"]
             [%])
          tokens))

;(defn merge-name-tags [tokens]

(defn mark-diffs [string]
  "Take two strings.
Iterate recursively through the longer (t-string) checking for matches.
If matched, consume and move on to the next one
If not matched, output marker instead, and don't consume a-string."
  (apply str (loop [t-str (clojure.string/join 
                            " " 
                            (tokenize 
                              (clojure.string/replace 
                                string "--" " -- ")))                             
                    a-str string 
                    c-str []]
               (if (empty? t-str)
                 c-str
                 (let [a1 (first a-str)
                       t1 (first t-str)]
                   (if (not (= a1 t1))
                     (if (= t1 \space)
                       (recur (rest t-str) 
                             a-str 
                             (conj c-str "|"))
                       (recur (rest t-str) 
                             (rest (rest a-str)) 
                             (conj c-str t1)))
                     (recur (rest t-str) 
                           (rest a-str) 
                           (conj c-str t1))))))))
 
(defn create-token-training-file [source destination]
  "Load a document, output as one sentence per line."
  (spit destination
        (apply str (mapcat #(apply str % "\r\n")
                           (map #(clojure.string/replace % "|" "<SPLIT>")
                                (map #(mark-diffs %)
                                     (mapcat #(get-sentences %)
                                             (just-get-sentences 
                                               source))))))))

(defn create-paragraph-training-file [source destination]
  "Load a document, output as one sentence per line."
  (spit destination
        (apply str (mapcat #(apply str % "\r\n")
                           (just-get-sentences 
                               source)))))

(defn create-sentence-training-file [source destination]
  "Load a document, output as one sentence per line."
  (spit destination
        (apply str (mapcat #(apply str % "\r\n")
                           (mapcat #(get-sentences %)
                             (just-get-sentences 
                               source))))))

(defn create-name-training-file [source destination]
  "Load a document, tokenize it, output as one sentence per line."
  (spit destination
        (clojure.string/replace  
          (apply str 
                 (mapcat #(apply str (apply str (interpose " " %)) "\r\n")
                         (map #(mark-token is-name? %)
                              (map 
                                tokenize 
                                (mapcat #(get-sentences %)
                                        (just-get-sentences 
                                          source))))))
                   " <END> <START> " " ")))

;(create-name-training-file (slurp "texts\\cleaned\\gutenberg\\austen-persuasion.txt") "texts\\training\\persuasion-name.train")
;(create-name-training-file (slurp "texts\\cleaned\\pg42671_clean.txt") "texts\\training\\pride-name.train")
;(create-name-training-file (slurp "texts\\cleaned\\gutenberg\\austen-emma.txt") "texts\\training\\emma-name.train")

;(create-paragraph-training-file
;  (slurp "texts\\training\\gutenberg-source-text.train")
;  "texts\\training\\gutenberg-paragraphs.train")

(comment
  (spit "texts\\training\\gutenberg-source-text.train"
  (apply str
         (slurp "texts\\cleaned\\pg120.txt")
         (slurp "texts\\cleaned\\pg228.txt")
         (slurp "texts\\cleaned\\pg1400.txt")
         (slurp "texts\\cleaned\\pg98.txt")
         (slurp "texts\\cleaned\\pg768.txt")
         (slurp "texts\\cleaned\\pg84.txt")
         (slurp "texts\\cleaned\\pg2591.txt")
         (slurp "texts\\cleaned\\pg1661.txt")
         (slurp "texts\\cleaned\\pg3177.txt")
         (slurp "texts\\cleaned\\pg3176.txt")
         (slurp "texts\\cleaned\\pg142.txt")
         (slurp "texts\\cleaned\\pg41093.txt")
         (slurp "texts\\cleaned\\pg41667.txt")
         (slurp "texts\\cleaned\\pg44133.txt")
         (slurp "texts\\cleaned\\gutenberg\\melville-moby_dick.txt")
         (slurp "texts\\cleaned\\pg42671_clean.txt")
         (slurp "texts\\cleaned\\gutenberg\\austen-emma.txt")
         (slurp "texts\\cleaned\\gutenberg\\carroll-alice.txt")
         (slurp "texts\\cleaned\\gutenberg\\chesterton-thursday.txt")
         ))  )

(comment
(create-sentence-training-file
  (apply str
         (slurp "texts\\cleaned\\pg120.txt")
         (slurp "texts\\cleaned\\pg228.txt")
         (slurp "texts\\cleaned\\pg1400.txt")
         (slurp "texts\\cleaned\\pg98.txt")
         (slurp "texts\\cleaned\\pg768.txt")
         (slurp "texts\\cleaned\\pg84.txt")
         (slurp "texts\\cleaned\\pg2591.txt")
         (slurp "texts\\cleaned\\pg1661.txt")
         (slurp "texts\\cleaned\\pg3177.txt")
         (slurp "texts\\cleaned\\pg3176.txt")
         (slurp "texts\\cleaned\\pg142.txt")
         (slurp "texts\\cleaned\\pg41093.txt")
         (slurp "texts\\cleaned\\pg41667.txt")
         (slurp "texts\\cleaned\\pg44133.txt")
         (slurp "texts\\cleaned\\gutenberg\\melville-moby_dick.txt")
         (slurp "texts\\cleaned\\pg42671_clean.txt")
         (slurp "texts\\cleaned\\gutenberg\\austen-emma.txt")
         (slurp "texts\\cleaned\\gutenberg\\carroll-alice.txt")
         (slurp "texts\\cleaned\\gutenberg\\chesterton-thursday.txt")
         )
  "texts\\training\\gutenberg-sentence.train")
)


(comment
  (create-name-training-file (slurp "texts\\cleaned\\pg42671_clean.txt") "texts\\training\\pg42671_name.train")
  ;(create-sentence-training-file (slurp "texts\\cleaned\\pg42671_clean.txt") "texts\\training\\pg42671_sentence.train")
  ;(create-token-training-file (slurp "texts\\cleaned\\pg42671_clean.txt") "texts\\training\\pg42671_token.train")

  (create-token-training-file
    (apply str
           (slurp "texts\\cleaned\\pg120.txt")
           (slurp "texts\\cleaned\\pg1400.txt")
           (slurp "texts\\cleaned\\pg98.txt")
           (slurp "texts\\cleaned\\pg768.txt")
           (slurp "texts\\cleaned\\pg84.txt")
           (slurp "texts\\cleaned\\pg2591.txt")
           (slurp "texts\\cleaned\\pg1661.txt")
           (slurp "texts\\cleaned\\pg3177.txt")
           (slurp "texts\\cleaned\\pg3176.txt")
           (slurp "texts\\cleaned\\pg142.txt")
           (slurp "texts\\cleaned\\pg41093.txt")
           (slurp "texts\\cleaned\\pg41667.txt")
           (slurp "texts\\cleaned\\pg44133.txt")
           (slurp "texts\\cleaned\\gutenberg\\melville-moby_dick.txt")
           (slurp "texts\\cleaned\\pg42671_clean.txt")
           (slurp "texts\\cleaned\\gutenberg\\austen-emma.txt")
           (slurp "texts\\cleaned\\gutenberg\\carroll-alice.txt")
           (slurp "texts\\cleaned\\gutenberg\\chesterton-thursday.txt")
           )
    "texts\\training\\gutenberg_token.train")
);end comment

;(def token-model (opennlp.tools.train/train-tokenizer "en" "texts\\training\\gutenberg-token.train" 600 5))
;(opennlp.tools.train/write-model token-model "texts\\training\\en-gutenberg-base-token.bin")
;(def namefinder-model (opennlp.tools.train/train-name-finder "texts\\training\\pride-name-1.train"))
;(opennlp.tools.train/write-model namefinder-model "texts\\training\\pride-name-1.bin")
  
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;
;; Analysis
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn process-sentence [text]
  (let [c-text text]
    c-text))


; iterate through sentence
; if this is a preposition or a proper noun, mark which character it belongs to.
;  if we have no character yet, mark CHAR=0
;  if we have a character, and this is immidately after, see if it's part of the same name, if so discard it.
;  if we have a character see if this is a reference to that character Mr. Darcy -> He or a new character.
;  if this is a new character, mark as CHAR+1, repeat

(defn is-char-token? [token]
  (if (empty? token)
    false
    (let [s (second token)]
      (cond
        (= s "NNP") true
        (= s "PRP") true
        :else false))))
 

(defn mark-characters [sentence-text]
  (loop [tagged-text sentence-text
         char ""
         last-token [nil nil]
         output []]
    (if (empty? tagged-text)
      output
      (let [t# (first tagged-text)
            p (if (is-char-token? last-token)
                (if (is-char-token? t#)
                  nil ;discard current
                  (first t#))
                (if (is-char-token? t#)
                  (second t#)
                  (first t#)))
            o (if (nil? p) output (conj output p))]
        (recur (rest tagged-text) char t# o)))))

(defn find-chars [sentence]
  (loop [text sentence
         char ""
         last-match [nil nil]
         output []]
    (if (empty? text)
      output
      (let [t (first text)
            m (if (nil? (second t))
                nil
                (re-matches #"(NNP)" (second t)))
            p (if (and (not (nil? m)) (= (second last-match) m))
                (apply str (interpose " " [(first last-match) (first t)]))
                (if (not (nil? m))
                  (first t)
                  nil))
            o (if (nil? p) output (conj (butlast output) p))]
        (recur (rest text) char [p m] o)))))
                  
(pos-filter names-filter #"(NNP|PRP)")

(defn catalog-names [source-text]
  (let [text (map #(-> % tokenize pos-tag) source-text)]
     (distinct (mapcat find-chars text))))

(defn catalog-names [source-text]
  (let [text (map #(-> % tokenize name-find) source-text)]
    text))
    
     ;(distinct (mapcat find-chars text))))

     ;(name-find
     
     
    ; ))
  
;(spit "texts\\output\\catalog-names.txt"
;     (apply str (interpose "\r\n"
;    (map #(apply str %)     
;    (catalog-names
;                         (get-sentences-from-text 
;                      (slurp "texts\\cleaned\\pg42671.txt"))))))) 

(pprint
  (let [sentences (get-sentences-from-text 
                    (slurp "texts\\cleaned\\pnp_excerpt.txt"))
        t-text (map #(-> % tokenize pos-tag) sentences)
        output (map mark-characters t-text)
        ]
    output))





;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Side Effects
;;;
;;; Output that happened to happen along the way to trying to get something else
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn output-action-edition [source destination]
  (spit destination
        (apply str 
               (map append-space
                    (map break-on-pilcrow
                         (get-sentences-of-category
                           (append-pilcrow
                             (paragraph-to-typed-sentences
                               (categorize-text 
                                 (clojure.string/split 
                                   (strip-italics 
                                     (strip-linebreaks 
                                       (clojure.string/replace 
                                         (slurp source) #"\r\n\r\n" "¶X¶X")))
                                   #"X¶X"))))
                           :action))))))
                           
(defn pnp-action-edition []
  (output-action-edition "texts\\cleaned\\pg42671.txt" "texts\\output\\pnp_action_edition.txt"))

(comment
 (pnp-action-edition))



































(comment
 (pprint
   (map noun-phrases
        (map chunker
             (map pos-tag
                  (map tokenize
                       (mapcat get-sentences
                               (just-get-sentences
                                 (slurp "texts\\cleaned\\pg42671.txt")
                                 ))))))))
                  
                  
;                  '(["With"  "the"  "Gardiners"  ","  "they"  "were"  "always"  "on"
;  "the"  "most"  "intimate"  "terms"  "."] ["Darcy"  ","  "as"  "well"
;  "as"  "Elizabeth"  ","  "really"  "loved"  "them"
;  ";"  "and"  "they"  "were"  "both"  "ever"
;  "sensible"  "of"  "the"  "warmest"  "gratitude"  "towards"
;  "the"  "persons"  "who"  ","  "by"  "bringing"  "her"  "into"
;  "Derbyshire"  ","  "had"  "been"  "the"  "means"  "of"  "uniting"
;  "them"  "."])))))



;(pprint
  ;(apply str (mapcat #(apply str % "\r\n")
  ;(map pos-tag
;        (map tokenize
;             (just-get-sentences 
;               (slurp "texts\\cleaned\\pg42671.txt"))));)

;(pprint
;  (noun-phrases
;   (chunker 
;     (pos-tag 
;       (tokenize (slurp "texts\\cleaned\\pg42671.txt"))))))








(defn create-action [sentence]
  "Takes an action sentence and converts it to a valid action-sentence-function that can be called by the characters."
  )

(defn display-action [action]
  "Takes a formatted action-sentence-function and runs it, returning a string that can be printed."
  )

(defn find-in-phrase [phrase regex]
  (some #(re-find regex %) (:phrase phrase)))

(defn variablize-phrase [phrase]
  (if (= (:tag phrase) "NP")
    (cond
      (find-in-phrase phrase #"\b(She|she)\b" ) 
      (assoc phrase :phrase ["[ACTOR-NAME-FEMALE]"])
      (find-in-phrase phrase #"\b(He|he)\b" ) 
      (assoc phrase :phrase ["[ACTOR-NAME-MALE]"])
      (find-in-phrase phrase
                      #"(Elizabeth|Lizzy|Lydia|Lady|Lucas|Kitty|Catherine|Miss|Maria|Mary|Jane|Mrs.|Miss)") 
      (assoc phrase :phrase ["[ACTOR-NAME-FEMALE]"])
      (find-in-phrase phrase
                      #"(Mr.|Bingley|Darcy|Hurst|William)") 
      (assoc phrase :phrase ["[ACTOR-NAME-MALE]"])
      
      :else phrase)
    phrase))
  
(defn substitute-noun-phrases [chunked-sentence]
  (map #(variablize-phrase %) chunked-sentence))

(defn example-action [character-one character-two]
  {:text (str " " " ") :characters [character-one character-two]})
 
(defn test-action-processing []
  (map append-space
       (map break-on-pilcrow
           (get-actions (slurp "texts\\cleaned\\pg42671.txt")))))

;(spit "texts\\output\\test.txt" (apply str (interpose "" (test-action-processing))))

;(with-open [wrtr (writer "texts\\output\\test.txt")]
;  (write
;    (apply str (interpose " " (test-action-processing)))
;    :pretty false
;    :stream wrtr))

;(pprint (test-action-processing))

(defn create-actor [name gender]
  {:name name :gender gender})

(create-actor "Mr. Darcy" :male)
(create-actor "Elizabeth" :female)

(defn example-action [actor]
  (apply str 
         (detokenize
         (tokenize "An example sentence about [ACTOR-NAME]."))
         ))
  
;(pprint (example-action {:name "Me"}))
  


;; Old scratchspace...
(comment
(binding [*print-right-margin* nil]
  (pprint
    ;(map #(detect-exposition-in-paragraph %)
    (get-sentences-of-category
         (paragraph-to-typed-sentences
           (categorize-text 
            (input-source-text-directly
              (slurp "texts\\cleaned\\pnp_excerpt.txt"))))
         :action)))
)
            
(comment   
  (defn test-pnp-processing []
    (with-open [wrtr (writer "texts\\output\\test2.txt")]
      (write
        ;(sentences-from-paragraphs
          ;(map paragraph-to-sentences 
               ;(filter #(= (:category %) :action)
                       (categorize-text
                         (input-source-text-from-file
                          "texts\\cleaned\\pnp_excerpt.txt"))
        :pretty true
        :stream wrtr)))
  (test-pnp-processing))
