(use 'clojure.pprint)
(use 'opennlp.nlp)
(use 'opennlp.treebank)
(use 'opennlp.tools.filters)

(def tokenize (make-tokenizer "models/en-token.bin"))
(def pos-tag (make-pos-tagger "models/en-pos-maxent.bin"))
(def chunker (make-treebank-chunker "models/en-chunker.bin"))

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

(comment
(pprint
  (filter (fn [t#] (re-find #"^NP$" (:tag t#)))
          (remove #(nil? (:tag %))
                  (chunker 
                    (pos-tag 
                      (tokenize "And when the party entered the assembly room, it consisted of only five
altogether; Mr. Bingley, his two sisters, the husband of the eldest, and
another young man."))))))
)

(pprint 
  (chunker 
      (pos-tag 
        (tokenize "And when the party entered the assembly room, it consisted of only five
altogether; Mr. Bingley, his two sisters, the husband of the eldest, and
another young man."))))

(pprint
  (fixed-noun-phrases
   (chunker 
     (pos-tag 
        (tokenize "And when the party entered the assembly room, it consisted of only five
altogether; Mr. Bingley, his two sisters, the husband of the eldest, and
another young man.")))))



(pprint
  (pos-tag 
    (tokenize "And when the party entered the assembly room, it consisted of only five
altogether; Mr. Bingley, his two sisters, the husband of the eldest, and
another young man.")))

;(pprint (noun-phrases
;          '({:phrase ["And"], :tag nil})))

(comment
 {:phrase ["when"], :tag "ADVP"}
 {:phrase ["the" "party"], :tag "NP"}
 {:phrase ["entered"], :tag "VP"}
 {:phrase ["the" "assembly" "room" ","], :tag "NP"}
 {:phrase ["it"], :tag "NP"}
 {:phrase ["consisted"], :tag "VP"}
 {:phrase ["of"], :tag "PP"}
 {:phrase ["only" "five" "altogether" ";"], :tag "NP"}
 {:phrase ["Mr." "Bingley" ","], :tag "NP"}
 {:phrase ["his" "two" "sisters" ","], :tag "NP"}
 {:phrase ["the" "husband"], :tag "NP"}
 {:phrase ["of"], :tag "PP"}
 {:phrase ["the" "eldest" "," "and"], :tag "NP"}
 {:phrase ["another" "young" "man"], :tag "NP"})



