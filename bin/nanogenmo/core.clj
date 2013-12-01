(ns nanogenmo.core
  (:require [clojure.pprint]
            [opennlp.nlp]
            [opennlp.treebank]
            [opennlp.tools.filters]
            [opennlp.tools.train]
            [clj-wordnet.core]
            )
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

(def wordnet (clj-wordnet.core/make-dictionary "texts/dict/"))

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
    (re-find #"^[.*]$" source-text) :annotation
    (re-find #"^CHAPTER" source-text) :annotation
    (= (clojure.string/upper-case source-text) source-text) :annotation
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

  
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;
;; Source Data
;;
;; Data other than the base text files.
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-data-file [data-file] 
  (clojure.string/split (first
    (input-source-text-directly (slurp data-file)))
  #" "))
  
;; TODO: add the frequency to the names and sort the list by it...
(defn census-name-list []
  "Returns a lazy-seq containing names derived from US Census."
  (let [male (map #(hash-map :word % :gender :masculine) 
                    (get-data-file "texts\\names\\male.first.txt"))
        female (map #(hash-map :word % :gender :feminine) 
                      (get-data-file "texts\\names\\female.first.txt"))]
    (concat female male)))
        ;(interleave male female)))))
  
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
    (re-find #"^(Queen|Countess|Duchess|Viscountess|Lady|Dowager|Mrs.|Mrs|Miss)$" token) true
    (re-find #"^(King|Duke|Admiral|Captain|Colonel|Viscount|Hon.|Lord|Sir|Mr.|Mr|St.|Doctor|Dr.|Dr)$" token) true
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

(defn process-for-analysis [sentence]
  "Take an action sentence and clean it up to be analyzed."
  )

(defn count-words [string]
  "given a string, count the number of occurances of each word."
  )







;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;
;; Names
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def example-name-list 
  [{:name "Elizabeth" :gender :feminine}
   {:name "Mr. Darcy" :gender :masculine}
   {:name "Emma" :gender :feminine}
   {:name "Mr. Bennet" :gender :masculine}
   {:name "Mr. Bingley" :gender :masculine}
   {:name "Mrs. Bennet" :gender :feminine}
   {:name "Mary" :gender :feminine}
   {:name "Lady Lucas" :gender :feminine}
   {:name "Sir William" :gender :masculine}
   {:name "Netherfield" :gender :place}
   {:name "London" :gender :place}
   {:name "Mr. Darcy" :gender :masculine}
   {:name "Derbyshire" :gender :place}
   {:name "Miss Bingley" :gender :feminine}
   {:name "Elizabeth" :gender :feminine}
   {:name "Jane" :gender :feminine}
   {:name "Lydia" :gender :feminine}
   {:name "England" :gender :place}
   {:name "Mr." :gender :masculine}
   {:name "Mrs. Hurst" :gender :feminine}
   {:name "Netherfield House" :gender :place}
   {:name "Bingley" :gender :masculine}
   {:name "Darcy" :gender :masculine}
   {:name "Meryton" :gender :place}
   {:name "Miss Bennet" :gender :feminine}
   {:name "Lucas Lodge" :gender :place}
   {:name "James" :gender :masculine}
   {:name "Miss" :gender :feminine}
   {:name "Miss Lucas" :gender :feminine}
   {:name "Sir William Lucas" :gender :masculine}
   ])


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
                (re-matches #"(NNP|NNPS)" (second t)))
            p (if (and (not (nil? m)) (= (second last-match) m))
                (apply str (interpose " " [(first last-match) (first t)]))
                (if (not (nil? m))
                  (first t)
                  nil))
            o (if (nil? p) output (conj (butlast output) p))]
        (recur (rest text) char [p m] o)))))
                  
(pos-filter names-filter #"(NNP|PRP)")

(defn catalog-names-1 [source-text]
  (let [text (map (fn [x] 
                                  ;{:pre [(string? x)]}
                                  (if (string? x)
                                    (-> x tokenize pos-tag)
                                    (print "name error:" x)
                                    )) 
                                
                  source-text)]
    
    (distinct
        (mapcat find-chars text))))

(defn catalog-names-2 [source-text]
  (let [text (map (fn [x] (if (string? x)
                            (-> x tokenize name-find))) source-text)]
    (distinct
      (mapcat concat
              (remove #(empty? %)
                      text)))))

; Take a sentence, mark the actors in it.
; Take an actor-marked sentence, add new actors. 

(defn genderize-name [name name-list]
  (let [n (filter #(= (:name %)) name-list)
        g (if (seq? n)
            (:gender (first n))
            false)]
    g))

(defn determine-name-gender [text-name-token]
  [(second text-name-token)
   (cond 
     (re-matches #"(PRP|PRP\$)" (second text-name-token))
     (cond
       (re-matches #"(I|me|my|mine|myself)" (first text-name-token))
       :first-person
       (re-matches #"(you|your|your|yourself)" (first text-name-token))
       :second-person
       (re-matches #"(she|her|hers|herself)" (first text-name-token))
       :feminine
       (re-matches #"(he|him|his|himself)" (first text-name-token))
       :masculine
       (re-matches #"(they|them|their|theirs|themself|themselves|we|us|our|ourselves)" (first text-name-token))
       :general
       :else :general)     
     :else (genderize-name (first text-name-token) example-name-list))
   (first text-name-token)
   ])
  
  ;(cond 
  ;  (re-matches #"PRP" (second text-name-token))
  

(defn is-actor-name? [text-token]
  (if (not (and (second text-token) (string? (second text-token))))
    false
    (let [r (re-matches #"(NNP|NNPS|PRP|PRP\$)" (second text-token))
          y (if (and r (string? (first text-token)))
              ;(if (name-find text-token)
                ;(second text-token)
                text-token
                false)]
      y)))

(defn annotate-names [source-text]
  "Take a sentence and return the same sentence with likely names tagged."
  (loop [text source-text
         output []]
    (if (empty? text)
      output
      (let [token (first text)
            match (re-matches #"(NNP|NNPS)" (second token))
            out (if match
                  (conj output "<START>" (first token) "<END>")
                  (conj output (first token)))]
        (recur (rest text) out))
      )))
  

(defn mark-actors [source-text]
  (loop [text source-text
         current-char nil
         last-token [nil nil nil] 
         output []]
    (if (empty? text)
      output
      (let [token (first text)
            match (is-actor-name? token)
            name-gender (if match (determine-name-gender token) nil)
            merge-with-last (is-actor-name? last-token)
            c (if (and (not (nil? current-char)) (= name-gender current-char))
                current-char
                name-gender); (inc (second current-char))])
            p (if name-gender name-gender (first token))
            out (if (and match merge-with-last)
                  (conj (vec (butlast output)) p)
                  (conj output p))]
        (recur (rest text) c token out))
      )))


;(defn pick-pronoun [gender pronoun]
;  (cond
;    (= "PRP" pronoun)
;    (cond 
;      (= gender :masculine) "he"
;      (= gender :feminine) "she"
;      :else "they"
;    (= "PRP$" pronoun) 
;  )

;; take a sentence
;; figure out who the characters are in it
;; substitute the new characters
;; return the new sentence

(defn substitute-characters [sentence char-list]
  (loop [text sentence
         last-char nil
         chars char-list
         output []]
    (if (empty? text)
      output
      (if (vector? (first text))
        (let [t (first text)
              same-char (cond
                          (= last-char nil) false
                          (= last-char "NNP") false                          
                          (= last-char "PRP") true ; assume pronouns refer to last named character...for now.
                          :else false                         
                          )
              c (if same-char last-char (first t))
              cl (if same-char chars (rest chars))
              w t
              ;(if same-char
              ;(cond
              ;  (re-matches #"(NNP|NNPS)" (first t))
              ;  (:name (first cl))
              ;  (re-matches #"(PRP|PRP\$)" (first t))
              ;  (pick-pronoun (:gender (first cl)) (first t))
              ]
          (recur (rest text) last-char cl (conj output w)))
        (recur (rest text) last-char chars (conj output (first text))))))) 

(defn enumerate-characters-in-sentence [sentence char-list]
  ;(detokenize
    (loop [text sentence
           char-tag nil
           char-count 0
           output []]
      (if (empty? text)
        output
        (if (vector? (first text))
          (let [t (first text)
                same-char (cond
                            (or (= char-tag :first-person) (= (second t) :first-person)) (= char-tag (second t))
                            (or (= char-tag :second-person) (= (second t) :second-person)) (or (= char-tag (second t) (= char-tag :general) (= (second t) :general)))
                            (= char-tag :general) true
                            (= char-tag (second t)) true
                            (= (second t) :general) true
                            :else false);;(= char-tag (second t)) 
                c (second t)
                count (if same-char (inc char-count) char-count)
                w count
                ]
            (recur (rest text) c count (conj output w)))
          (recur (rest text) char-tag char-count (conj output (first text)))))))


(defn process-data [source-text]
  (map #(-> % tokenize pos-tag mark-actors) source-text))

(defn generate-recharacterization [source-text]
  (let [name-list (catalog-names-1 source-text)
        action-list (process-data source-text)]
    (map #(substitute-characters % name-list) action-list)
    ))

(defn annotate-names-in-text [source-text]
  (map #(-> % tokenize pos-tag annotate-names detokenize) source-text)) 


(comment  
(spit "texts\\training\\gutenberg-annotate-names.train"
      (apply str (interpose "\r\n"
                            (annotate-names-in-text
                              (get-sentences-from-text 
                                  (slurp  "texts\\training\\gutenberg-source-text.train"))))))
)

(comment
(pprint
  (generate-recharacterization
     (get-sentences-from-text 
      (slurp "texts\\cleaned\\pnp_excerpt.txt"))))
)

     ;(distinct (mapcat find-chars text))))

     ;(name-find
     
     
    ; ))

(comment
(spit "texts\\output\\gutenberg-catalog-names-2.txt"
    (apply str (interpose "\r\n"
                         (catalog-names-2
                             (get-sentences-from-text 
                               (slurp  "texts\\training\\gutenberg-source-text.train"))))))
)



(comment
(pprint
 (let [sentences (get-sentences-from-text 
                   (slurp "texts\\cleaned\\pnp_excerpt.txt"))
       t-text (map #(-> % tokenize pos-tag) sentences)
       output (map mark-characters t-text)
       ]
   output)))



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;
;; Chunking for name subsitiution
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn is-phrase-name? [chunk]
  (if (empty? (name-find (:phrase chunk)))
    false
    true
    ))
  
(defn gen-output-action [action names]
  "Take a chunked sentence, and walk through it, outputting as an action..."
  (loop [act action
         last-char nil
         name-list (shuffle names) 
         output []]
    (if (empty? act)
      output
      (let [t (first act)
            c (if (= (:tag t) "NP") ; is this a name?
                (is-phrase-name? t)
                false)
            same-char (if (nil? last-char);what name do we use?
                        false        
                        )
            n-list (if same-char name-list (rest name-list))
            n (first n-list)
            cur-char last-char
            p (if c [n] (:phrase t))
            o (conj output p)]
        (recur (rest act) cur-char name-list o)))))

(defn output-action [action]
  (clojure.string/join [
         (detokenize (into [] (mapcat concat action)))
         "."]))

(defn swap-names [source-text]
  (let [name-list (catalog-names-1 source-text)
        action-list (map #(apply str (clojure.string/join " " (tokenize %))) source-text)
        chunked-list (map #(-> % tokenize pos-tag chunker) source-text)
        ;parsed-list (map #(array-map :parsed (parser [%]) :text %) action-list)
        ;filtered-parsed (map #( (:parsed %))
        made-actions (map #(gen-output-action % name-list) chunked-list)
        ]
    
    
    ;(map #(substitute-characters % name-list) action-list)
    ;(map #(detokenize (into [] (mapcat concat %))) made-actions)
    made-actions
    ))


(defn output-gutenberg-shuffle []
(spit "texts\\output\\gutenberg-shuffle.txt"
      (apply str 
             (mapcat concat
                     (let [actions ;(map #(apply str (concat % [["."]]))
                           (swap-names
                             (get-sentences-from-text 
                               (slurp  "texts\\training\\gutenberg-source-text.train")))]
                       (loop [count 0 output []]
                         (if (> count 5000)
                           (mapcat concat output)
                           (recur (inc count)
                                  (conj output 
                                        (conj
                                          (interpose " " 
                                             (map output-action 
                                                  (take (+ 1 (rand-int 7)) (shuffle actions))))
                                          "\r\n\r\n")
                                        ))))))))
        ) 

;(let 

;(spit "texts\\output\\shuffled-5.txt"
;      (doall
;        (let [actions (swap-names
;                        (get-sentences-from-text 
;                        (slurp "texts\\training\\gutenberg-source-text.train")))]
;          (doall (take 200 (doall
;                         (repeatedly
;                           #(apply str (doall (take 5 (shuffle actions)))
;                                  "\r\n"))))))))
  
;(pprint                          
  ;(map output-action 
  ;     (shuffle
  ;       (swap-names
  ;         (get-sentences-from-text 
  ;           (slurp "texts\\training\\gutenberg-source-text.train"))))))


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


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;
;; Book output
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn print-debug [text body]
  (pprint text)
  body)

(defn get-data [source-text]
  (paragraph-to-typed-sentences
    (categorize-text 
      (map #(clojure.string/replace % "`" "'")
           (input-source-text-directly
             source-text)))))

(defn name-census [word]
  (let [matches (filter 
                  #(re-find (re-pattern (:word %))
                            word) 
                  (census-name-list))]
    (if (or (empty? matches) (nil? matches))
      :unknown
      (:gender (first matches)))))

(defn guess-gender [word]
  "Takes a noun or pronoun and tries to guess the grammatical gender."
  (cond 
    (re-matches #"(I|me|my|mine|myself)" word) :first-person
    (re-matches #"(You|Your|Yourself|you|your|your|yourself)" word) :second-person
    (re-matches #"(She|Herself|Hers|Her|she|her|hers|herself)" word) :feminine
    (re-matches #"(He|Him|His|Himself|he|him|his|himself)" word) :masculine
    (re-matches #"(They|Them|Their|Theirs|Themself|Themselves|We|Us|Our|Ourselves|they|them|their|theirs|themself|themselves|we|us|our|ourselves)" word) :group
    (re-matches #"(It|it|Itself|itself)" word) :neuter
    (re-find #"(Baroness|Godess|godess|Queen|Countess|Duchess|Viscountess|Lady|Dowager|Mrs\.|Mrs |Miss)" word) :feminine
    (re-find #"(Baron|God|god|King|Duke|Admiral|Captain|Colonel|Viscount|Hon\.|Lord|Sir|Mr\.|Mr |St\.|Doctor|Dr\.|Dr )" word) :masculine
    
    ;(re-find #"(Bennets|Lucases|Harvilles|Gardiners|Collinses|Ibbotsons|Durands|Musgroves)" word) :group
    ;(re-find #"(Elizabeth|Catherine|Charlotte|Caroline|Georgina|Henrietta|Lousia|Penelope|Kitty|Eleanor|Elinor|Lizzy|Lydia|Louisa|Alicia|Maria|Jemima|Sarah|Jane|Anne|Harriet|Mary|Pen|Anna|Anne)" word) :feminine
    ;(re-find #"(Christopher|Edmund|Edward|Walter|Fredrick|Fredric|William|George|James|Harry|Henry|Charles|Basil|Dick|John)" word) :masculine
    ;(re-find #"^(Musgrove|Wentworth|Harville|Hayter|Trent|Brigden|Musgrove|Benwick|Bennet|Bingley|Lucas|Carteret|Hurst|Smith|Darcy|Gardiner|Wickham|Collins|Rooke|Wallis|Fitzwilliam|de|Bourgh|Elliot|Russell|Shepherd|Clay|Long|Frankland|Morley|Metcalfe|Nicholls|Phillips|Pope|Pratt|Reynolds|Robinson|Stone|Watson|Webbs|Younge|Jenkinson|Jones|King|Long|Goulding|Grierson|Grantley|Haggerston|Harrington|Annesley|Chamberlayne|Dawson|Denny|Forster|Croft|Brand|Atkinson|Mackenzie|Hamilton|Dalrymple|Speed|Maclean|Ives|Spicer|Shirley|Morris)$" word) true
    :else (name-census word) 
  ))

(defn merge-gender [g1 g2]
  (if (nil? g1) 
    g2
    (if (nil? g2)
      g1
      (if (= :unknown g1)
        g2
        g1))))

(defn token-is-name? [token]
  :doc "Takes a POS-tagged token and returns true is the token is probably a name."
  (cond
    (re-matches #"(NNP)" (:tag token)) true
    ;(re-matches #"(PRP|PRP\$)" (:tag token)) true
    :else false))

(defn combine-name-vectors [tokenized-sentence]
  (let [combined 
        (reduce #(cond
                   (and (map? %1) (map? %2))
                   (flatten (concat [(merge %1 
                                            (merge %2 
                                                   {:gender (merge-gender (:gender %1) (:gender %2))
                                                    :word 
                                                    (clojure.string/join " " [(:word %1)
                                                                              (:word %2)])}) )]))      
                   (and (map? (last %1)) (map? %2))
                   (flatten (concat [(butlast %1) 
                                     (merge (last %1) 
                                            (merge %2 
                                                   {:gender (merge-gender (:gender (last %1)) (:gender %2))
                                                    :word 
                                                    (clojure.string/join " " [(:word (last %1))
                                                                              (:word %2)])}) )]))
                   :else (flatten (concat [%1 %2]))
                   )
                tokenized-sentence)]
    (remove nil? combined)))

(defn mark-names [tokenized-sentence]
  "Take a pos-tagged, tokenized sentence, and return it with the names replaced with functions."
  (combine-name-vectors
      (map #(if (token-is-name? %) % (:word %)) ;filter out names
           (map #(hash-map :word (first %)
                       :tag (second %)
                      :gender (guess-gender (first %))
                      )
                tokenized-sentence))))

(defn translate-pronouns [word]
  (cond
    (= (:tag word) "NNP") "NNP"
    (= (:tag word) "PRP") "PRP"
    :else (:word word)))


(defn number-characters [sentence]
  "Take a processed sentence and mark which character is associated with each noun and pronoun."
  (loop [input sentence
         output []
         char-count 0
         m-count 0
         f-count 0
         n-count 0
         last-char nil
         ]
    (if (empty? input)
      output
      (let [word (first input)
            m? (map? word)
            n? (if m? 
                 (= (:tag word) "NNP") 
                 false)
            chars (if n? (inc char-count) char-count)
            gender (:gender word)
            counting (merge {:count chars} 
                            (cond
                              (= gender :masculine) {:gender-count m-count}
                              (= gender :feminine) {:gender-count f-count}
                              (= gender :unknown) {:gender-count n-count}
                              :else {}))
            mc (if (= gender :masculine) (inc m-count) m-count)
            fc (if (= gender :feminine) (inc f-count) f-count)
            nc (if (= gender :unknown) (inc n-count) n-count)
            output-word (if n? (merge word counting) word)
            cur-char last-char
            ]
        (recur (rest input) (conj output output-word) chars mc fc nc cur-char)))))
  

(comment
(defn insert-characters [sentence char-list]
 (loop [input (:processed sentence)
        output []
        characters char-list
        last-char nil
        ]
   (if (empty? input)
     output
     (let [word (first input)
           name? (map? word)
           new-character? (if (map? word)
                            (cond
                              (= (:tag word) "NNP") true
                              (= (:tag word) "PRP") (if (nil? last-char)
                                                      true
                                                      false)
                              :else false)
                            false)
           c-list (if new-character? (rest characters) characters)
           current-char (if new-character? (first c-list) last-char)
           output-word (if (map? word)
                         (translate-pronouns word)
                         word)
           ]
       (recur (rest input) (conj output output-word) c-list current-char)))))
)

(defn random-character [characters]
  (apply str (:word (first (shuffle (characters))))))

;(defn random-word [text]
;  (time
;    ;  (last (sort-by count (clojure.string/split (apply str text) #" ")))))
;     (last (shuffle (tokenize (apply str text))))))

(defn longest-word [text]
  (time
    ;  (last (sort-by count (clojure.string/split (apply str text) #" ")))))
     (last (sort-by count (tokenize (apply str text))))))

;(defn insert-characters [sentence char-list]
;(map
; #(cond
   ;    (map? %) (first char-list)
   ;    :else %)
   ; (:processed sentence)))

(defn make-sentence [action character-list]
  (let [processed-action action        
        new-sentence
        (map
          (fn [token]
            (str
              (cond
                (string? token) token
                (map? token) (str 
                               (if (nil? (:word token)) 
                                 "ERROR: nil" 
                                 (if (or (nil? (:count token)) (nil? (:gender token)) (nil? (:gender-count token)))
                                   (str "ERROR: count <" token ">")
                                   (let [gender (:gender token)
                                         filtered (filter #(= (:gender %) gender) character-list)
                                         chosen-character (nth filtered 
                                                               (:gender-count token)
                                                               (str "ERROR: " filtered " - " {:word (str (:gender token))}))
                                         ]
                                     (if (:word chosen-character)
                                       (str (:word chosen-character))
                                       (str chosen-character));[(:gender token) (:gender-count token)])
                                         )
                                   )))
                :else (str "ERROR: <" token ">"))
              ))
              ;(if (map? token)
              ;  (str token)
                ;(cond
                ;  (:count token) ;output first matching character... 
                ;  (let [gender (:gender token)
                ;        filtered (filter #(= (:gender %) gender) character-list)
                ;        chosen-character (nth filtered (:gender-count token) {:word (str (:gender token))})]
                ;    (:word chosen-character))
                ;  (:word token) (:word token)          
                ;  :else token)
               ; token))) ;ordinary token
         processed-action)
        ;(map str action)
          ]
    ;(pprint new-sentence)
    (if (every? string? new-sentence)
      (detokenize new-sentence)
      (pprint "ERROR:" new-sentence))
      ))

(defn make-paragraph [actions characters]
  (apply str
         (clojure.string/join " "
                (map #(make-sentence % characters) 
                     (take (+ 1 (rand-int 7)) (shuffle actions))))
         "\r\n\r\n"))

(defn make-scene [action-list character-list] 
  "Output a number of paragraphs, based on the actions and characters provided."
  (time
  (apply str
         (let [chars (shuffle character-list)
               acts (take (int (/ (count action-list) 4)) action-list)]; take the first quarter of the actions and shuffle them
           (take (+ 1 (rand-int 7)) (repeatedly #(make-paragraph acts chars)))))))

(defn make-chapter [action-list character-list chapter-number]
  (time
  (apply str
  (let [acts (shuffle (take (int (/ (count action-list) 4)) action-list))
        body-text (take (+ 3 (rand-int 7))
                        (repeatedly #(make-scene acts character-list)))
        chapter-name (longest-word body-text)
        ]
    (print "Wrote Chapter " chapter-number ": " chapter-name "\n")
    (clojure.string/join ["\r\n##CHAPTER " chapter-number ": " (clojure.string/upper-case chapter-name) "\r\n\r\n" (apply str body-text) "\r\n\r\n"])
    ))))

(defn write-book [action-list character-list]
  (apply str
         (loop [itr 1 output []]
           (if (> itr 26)
             output
             (recur (inc itr) (concat output (make-chapter (shuffle action-list) character-list itr)) )))))

(defn strip-nils [sentences]
  (remove nil? sentences)
  )

(defn make-book [raw-source-text]
  (let [raw-text raw-source-text;(slurp "texts\\cleaned\\pnp_excerpt.txt")
        paragraphs (time (get-data raw-text))
        source-text (flatten (vals (mapcat :categorized paragraphs)))       
        character-name-list (time (print-debug "Characters" (remove nil? (map #(hash-map :word % :gender (guess-gender %)) 
                                                                              (distinct (concat 
                                                                                          (catalog-names-1 source-text) 
                                                                                          (catalog-names-2 source-text)))))))
        print-names (pprint character-name-list)
        action-sentences (time (print-debug "Actions" (filter #(not (nil? %)) (flatten (map #(:action (:categorized %)) paragraphs)))))
        actions-list
        (time  
              (pmap (fn [sen]
                      {:pre [(string? sen)]}
                      (let [tokenized (tokenize sen)
                            ;parsed (parser [(clojure.string/join " " tokenized)])
                            pos-tagged (pos-tag tokenized)
                            ;chunked (chunker pos-tagged)
                            processed (number-characters (into [] (mark-names pos-tagged)))
                            ]
                        (print ".")
                        ;(hash-map
                          ;:text %
                          ;:tokenized tokenized
                          ;:parsed parsed
                          ;:pos pos-tagged
                          ;:chunked chunked
                          processed
                          ))
                   action-sentences))
        body-text (write-book actions-list character-name-list)
        book-title (str (longest-word body-text)); (random-character character-name-list) (random-word body-text))
               ]
    (clojure.string/join ["#" (clojure.string/upper-case book-title) ": A Novel\r\n\r\n"
                          (apply str body-text)
                          "\r\n\r\n"])
    ;    (pprint source-text)                  
    ))


;(time 
;(spit "texts\\output\\book-test-2.txt"
;  (apply str (make-book (slurp "texts\\cleaned\\pnp_excerpt.txt")))))

(defn novel-source-list []
  [;"texts\\cleaned\\pnp_excerpt.txt"
   "texts\\cleaned\\pg42671_clean.txt"
  ; "texts\\cleaned\\pg142.txt"
  ;"texts\\cleaned\\pg41667.txt"
  ; "texts\\cleaned\\pg1400.txt"
  ; "texts\\cleaned\\pg98.txt"
  ; "texts\\cleaned\\pg768.txt"
  ; "texts\\cleaned\\pg2591.txt"
  ; "texts\\cleaned\\pg2776.txt"
  ; "texts\\cleaned\\pg215.txt"
  ; "texts\\cleaned\\pg910.txt"
  ; "texts\\cleaned\\pg13821.txt"
  ; "texts\\cleaned\\pg8183.txt"
  ; "texts\\cleaned\\pg5713.txt"
  ; "texts\\cleaned\\pg10806.txt"
  ; "texts\\cleaned\\pg7838.txt"
  ; "texts\\cleaned\\pg7477.txt"
  ; "texts\\cleaned\\pg13820.txt"
  ; "texts\\cleaned\\gutenberg\\austen-emma.txt"
  ; "texts\\cleaned\\gutenberg\\austen-persuasion.txt"
  ; "texts\\cleaned\\gutenberg\\austen-sense.txt"
  ; "texts\\cleaned\\gutenberg\\burgess-busterbrown.txt"
  ; "texts\\cleaned\\gutenberg\\chesterton-brown.txt"
  ; "texts\\cleaned\\gutenberg\\chesterton-thursday.txt"
  ; "texts\\cleaned\\gutenberg\\melville-moby_dick.txt"
  ; "texts\\cleaned\\pg2600.txt"
  ; "texts\\cleaned\\pg62.txt"
  ; "texts\\cleaned\\pg45.txt"
   ;"texts\\cleaned\\pg730.txt"
   ])

(defn input-source-text []
  (apply str
         (map slurp (novel-source-list))))

(defn make-novel []
  (time
    (spit "texts\\output\\novel.markdown"
          (apply str (make-book (input-source-text)))))
  (print "\nDone\n"))

(make-novel)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;
;; Main
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (make-novel))









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

;(create-actor "Mr. Darcy" :male)
;(create-actor "Elizabeth" :female)

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
