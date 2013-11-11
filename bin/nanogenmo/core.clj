(ns nanogenmo.core
  (:require [clojure.pprint]
            [opennlp.nlp]
            [opennlp.treebank]
            [opennlp.tools.filters])
  (:use [clojure.pprint]
        [opennlp.nlp]
        [opennlp.treebank]
        [opennlp.tools.filters])
  (:gen-class))

(use 'clojure.java.io)

(def get-sentences (make-sentence-detector "models/en-sent.bin"))
(def tokenize (make-tokenizer "models/en-token.bin"))
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

(fixed-chunk-filter fixed-noun-phrases #"^NP$")
;;;

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (println "Hello, World!"))

(defn strip-italics [text]
  "Removes _underscored italics_ from the text. It'd be nice to find a way to include these, later."
  (clojure.string/replace text "_" ""))

(defn strip-linebreaks [text]
  "Returns string with linebreaks stripped out."
   (clojure.string/replace 
     (clojure.string/replace text #"[\r\n]+" "_@_")
     "_@_" " "))
  
(defn mark-paragraphs [source-text]
  (clojure.string/replace source-text #"\r\n\r\n" "¶X¶X"))

(defn break-on-pilcrow [source-text]
  (clojure.string/replace 
      source-text 
      #"¶" "\r\n\r\n";(str \newline \newline)
      ;(clojure.string/re-quote-replacement \newline)
      ))

(defn get-paragraphs [source-text]
  "Returns source-texts broken into a list of paragraphs."
  (clojure.string/split source-text #"X¶X"))

(defn categorize-paragraph [source-text]
  "Returns a category based on the contents of the source-text: dialog, action, or exposition."
  (cond
    (re-find #"\"" source-text) :dialogue
    (re-find #",$" source-text) :ends-with-comma ;only last sentence is dialogue
    ;(re-find #"\b(She|she|He|he)\b" source-text) :action
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
  ;(assoc paragraph (hash-map :sentences (get-sentences (:text paragraph)))))

(defn paragraph-to-typed-sentences [paragraphs]
  (map #(assoc % :categorized 
               (let [sentences (get-sentences (:text %))
                     category (:category %)]
                 (cond 
                   (= category :dialogue) {:dialogue sentences}
                   (= category :action) {:action sentences}
                   (= category :ends-with-comma) {:action (butlast sentences) 
                                                  :dialogue (last sentences)}
                   )))
         paragraphs))

(comment
(defn mark-last-sentence [paragraph]
  (if (not (nil? paragraph))
    (let [sentences (:action (:categorized paragraph))]
      (if (not (nil? sentences))
        (assoc-in paragraph [:categorized :action] 
                  (assoc sentences 
                         (- (count sentences) 1) 
                         (str (last sentences) "-----------¶")))
        paragraph))
    paragraph))
  )

(defn sentences-from-paragraphs [paragraphs]
  "Given a collection of paragraphs, grab just the sentences."
  (mapcat #(:sentences %) paragraphs))

(defn grab-sentences-of-type [paragraphs sentence-type]
  (sentences-from-paragraphs
         (map paragraph-to-sentences 
              (filter #(= (:category %) sentence-type)
                      paragraphs))))

(defn get-sentences-of-category [paragraphs category]
  (mapcat #(category (:categorized %)) paragraphs))

(defn get-actions [source-text]
  (get-sentences-of-category
    (paragraph-to-typed-sentences
        (categorize-text 
          (input-source-text-directly
            source-text)))
    :action))

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

  ;(name-find (tokenize sentence)))
  ;(detokenize (dechunk 
  ;(substitute-noun-phrases 
  ;(chunker 
  ; (pos-tag (tokenize sentence)))));))

(defn example-action [character-one character-two]
  {:text (str " " " ") :characters [character-one character-two]})
  
  
(defn test-action-processing []
  (map break-on-pilcrow
      (get-actions (slurp "texts\\cleaned\\pg42671.txt"))))


(spit "texts\\output\\test.txt" (apply str (interpose " " (test-action-processing))))

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
