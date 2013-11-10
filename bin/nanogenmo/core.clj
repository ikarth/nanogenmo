(ns nanogenmo.core
  (:require [clojure.pprint]
            [opennlp.nlp]
            [opennlp.treebank])
  (:use [clojure.pprint]
        [opennlp.nlp]
        [opennlp.treebank])
  (:gen-class))

(use 'clojure.java.io)

(def get-sentences (make-sentence-detector "models/en-sent.bin"))
(def tokenize (make-tokenizer "models/en-token.bin"))
(def detokenize (make-detokenizer "models/english-detokenizer.xml"))
(def pos-tag (make-pos-tagger "models/en-pos-maxent.bin"))
(def name-find (make-name-finder "models/namefind/en-ner-person.bin"))
(def chunker (make-treebank-chunker "models/en-chunker.bin"))


(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (println "Hello, World!"))

;(defn input-source-text [source-text]
;  (with-open [rdr (reader source-text)]
;    (doseq [line (line-seq rdr)]
;      (println line))))

(defn strip-linebreaks [text]
  "Returns string with linebreaks stripped out."
  (clojure.string/replace text "\r\n" " "))

;(defn input-source-text-2 [source-text]
;  (get-sentences (slurp source-text)))

(defn mark-paragraphs [source-text]
  (clojure.string/replace source-text #"\r\n\r\n" "¶"))

(defn get-paragraphs [source-text]
  "Returns source-texts broken into a list of paragraphs."
  (clojure.string/split source-text #"¶"))

(defn categorize-paragraph [source-text]
  "Returns a category based on the contents of the source-text: dialog, action, or exposition."
  ;(clojure.string/
  )

(defn input-source-text [source-text]
  "Takes cleaned source text and formats it into useful paragraphs."
  (get-paragraphs (strip-linebreaks 
                    (mark-paragraphs 
                      (slurp source-text)))))

(input-source-text
  "texts\\cleaned\\pnp_excerpt.txt")


;(get-sentences (strip-linebreaks
;(get-paragraphs (strip-linebreaks (mark-paragraphs (slurp "texts\\cleaned\\pnp_excerpt.txt"))))

