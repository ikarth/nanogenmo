(defproject nanogenmo "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.5.1"]
                 [clj-diff "1.0.0-SNAPSHOT"]
                 [instaparse "1.0.1"]
                 ;[org.apache.opennlp/opennlp-tools "1.5.3"]
                 ;[jwnl "1.3.3"]
                 ;[org.apache.opennlp/opennlp-maxent "3.0.3"]
                 ;[opennlp-tools "1.5.3"]
                 [clojure-opennlp "0.3.1"]
                 ;[clojure-opennlp "0.3.2-SNAPSHOT"]
                 [clj-wordnik "0.1.0-alpha1"]
                 [clj-wordnet "0.0.5"]
                 ]
  :main ^:skip-aot nanogenmo.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
