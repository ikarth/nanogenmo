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
(defmacro fixed-chunk-filter
  "Declare a filter for treebank-chunked lists with the given name and regex."
  [n r]
  (let [docstring (str "Given a list of treebank-chunked elements, "
                       "return only the " n " in a list.")]
    `(defn ~n
       ~docstring
       [elements#]
       (filter (fn [t#] (re-find ~r (:tag t#))) 
               (remove #(nil? (:tag %)) elements#)))))

(chunk-filter fixed-noun-phrases #"^NP$")
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
  (get-paragraphs (strip-italics (strip-linebreaks 
                    (mark-paragraphs 
                       source-text)))))

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
    (re-find #"^(Captain|Colonel|Lord|Sir|Mr.|Mr)$" token) true
    (re-find #"^(Bennets|Lucases|Harvilles|Gardiners|Collinses)$" token) true
    (re-find #"^(Elizabeth|Catherine|Charlotte|Caroline|Georgina|William|Charles|Kitty|Walter|Lizzy|Lydia|Maria|Jane|Anne|Harriet|Mary|Pen)$" token) true
    (re-find #"^(Wentworth|Harville|Benwick|Bennet|Bingley|Lucas|Hurst|Smith|Darcy|Gardiner|Wickham|Collins|Fitzwilliam|de|Bourgh|Elliot|Russell|Shepherd|Clay|Long|Metcalfe|Nicholls|Phillips|Pope|Pratt|Reynolds|Robinson|Stone|Watson|Webbs|Younge|Jenkinson|Jones|King|Long|Goulding|Grantley|Haggerston|Harrington|Annesley|Chamberlayne|Dawson|Denny|Forster|Croft)$" token) true
    :else false))
;her Ladyship, the Archbishop, the Secretary, the Superior, Mother Superior, 
;Places: Rosings, London, Pemberley, Hertfordshire, Gracechurch-street, Brighton, Meryton, Derbyshire, Rosing's Park, Lambton, Hunsford Parsonage, Netherfield, 
;Punctuation: !-- -- ;-- .--

(defn mark-token [predicate? tokens]
  (mapcat #(if (predicate? %)
             ["<START>" % "<END>"]
             [%])
          tokens))

;(defn mark-string [item]
;  (mapcat #(cond
;             () 
;             :else item
;             )))

; , -> <SPLIT>,
; ; -> <SPLIT>;
; : -> <SPLIT>:
; -- -> <SPLIT>--<SPLIT>
; ' "' -> "<SPLIT>
; '" ' -> <SPLIT>"
; 

;; tokenize with existing model
;; compare against untokenized string
;; mark splits where they don't match

;(defn find-mismatches [a-string t-string]
;  "Take two strings. Find the point where they don't match. Mark it. Return string of equal length to t-string by padding a-string with split markers."
; )

;(defn mark-split-diff [a-string]
;  (let [t-string (apply str (tokenize a-string))
;        ]
;    
;    ))

;(defn mark-token-splits [string]
; (-> 
;   #(clojure.string/replace % "," "<SPLIT>,")
;   #(clojure.string/replace % ";" "<SPLIT>;")
;   #(clojure.string/replace % ":" "<SPLIT>:")
;   #(clojure.string/replace % "--" "<SPLIT>--<SPLIT>")
;   #(clojure.string/replace % " \"" "\"<SPLIT>")
;   #(clojure.string/replace % "\" " "<SPLIT>\"")
;   ))

;; Take two strings
;; walk through the longer (t-string) checking for matches
;; if matched, consume and move on to the next one
;; if not matched, output marker instead, and don't consume a-string
 
(defn mark-diffs [string]
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
                             (conj c-str "✗"))
                       (recur (rest t-str) 
                             (rest (rest a-str)) 
                             (conj c-str t1)))
                     (recur (rest t-str) 
                           (rest a-str) 
                           (conj c-str t1))))))))
 
(defn create-token-training-file [source destination]
  "Load a document, output as one sentence per line."
  (spit destination
        (apply str (mapcat #(apply str % "\n")
                           (map #(clojure.string/replace % "✗" "<SPLIT>")
                                (map #(mark-diffs %)
                                     (just-get-sentences 
                                       source)))))))

(defn create-sentence-training-file [source destination]
  "Load a document, output as one sentence per line."
  (spit destination
        (apply str (mapcat #(apply str % "\n")
               (just-get-sentences 
                    source)))))

(defn create-name-training-file [source destination]
  "Load a document, tokenize it, output as one sentence per line."
  (spit destination
        (apply str 
               (mapcat #(apply str (apply str (interpose " " %)) "\n")
                       (map #(mark-token brute-force-is-name? %)
                            (map 
                              tokenize 
                              (just-get-sentences 
                                     source)))))))

(create-name-training-file (slurp "texts\\cleaned\\gutenberg\\austen-persuasion.txt") "texts\\training\\persuasion_name.train")
(comment
(create-name-training-file (slurp "texts\\cleaned\\pg42671_clean.txt") "texts\\training\\pg42671_name.train")
(create-sentence-training-file (slurp "texts\\cleaned\\pg42671_clean.txt") "texts\\training\\pg42671_sentence.train")
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
)

(comment
  (def token-model (opennlp.tools.train/train-tokenizer "texts\\training\\gutenberg_base_token.train"))
  (opennlp.tools.train/write-model token-model "texts\\training\\gutenberg_base_token.bin")
  )

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
