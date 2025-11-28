(ns training.showcase
  (:gen-class)
  (:require [training.embedding :as mbd]
            [training.simple-perceptron :as network]))

(def mistypes ["Good jog, Aleksey"
               "God job, Aleksey"
               "Goob job, Aleksey"
               "Goodj ob, Aleksey"
               "Gudd job, Aleksey"
               "Good zob, Aleksey"
               "Gad job, Aleksey"
               "Grod job, Aleksey"
               "Bood job, Aleksey"
               "Foof job, Aleksey"
               "Yood job, Aleksey"
               "Hoox job, Aleksey"
               "Wook job, Aleksey"
               "Qood job, Aleksei"
               "Hood joab, Aleksey"
               "Cood job, Alksey"
               "Nood job, Aleskey"
               "Moos job, Alexey"
               "Soop job, Aleksay"
               "Boom job, Alexy"])

(def knowledge {1 "Good job, Aleksey"})

(defn train-network
  "Тренирует сеть целиком.
  Возвращает веса и смещения для каждого объекта в map"
  []
  (println "Training...")
  (map (fn [mistype] (let [reference (knowledge 1)
                           features (mbd/encode mistype reference)
                           initial-weights (vec (repeat mbd/feature-capacity 0))
                           initial-bias 1
                           evaluated-weights-bias (network/train features initial-weights 0.1 initial-bias 0 1)]
                       (hash-map :mistype mistype :weights-bias evaluated-weights-bias)))
       mistypes)
  )

(defn -main []
  (let [vector-of-maps (vec (train-network))]
    (doseq [map vector-of-maps] (println map))
    (println "Testing...")
    (doseq [map vector-of-maps]
      (let [[mistype [weights bias]] [(map :mistype) (vals (map :weights-bias))]
            features (mbd/encode mistype (knowledge 1))
            decoded (mbd/decode (network/activate-neuron features weights bias) knowledge)]
        (println (format "%s ==> %s", mistype decoded)))))
  )