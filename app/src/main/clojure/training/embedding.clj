(ns training.embedding
  (:require [clojure.string :as s]))

(defn tokenize
  "Чувствительно к регистру (специально в целях практики)"
  [text]
  (s/split text #""))

(def memo-tokenize (memoize tokenize))

(def feature-capacity "Длина вектор признаков, которую может принять нейрон" 20)

(defn pad-left-vec
  "Возвращает вектор, дополненный до указанной длины"
  [vector target filler]
  (let [diff (- target (count vector))]
    (if (pos? diff)
      (vec (concat vector (repeat diff filler)))
      (vec vector)))
  )


(defn encode
  "Переводит текст в единичный вектор"
  [text reference]
  {:pre [(string? text)
         (string? reference)]
   }
  (let [input-tokens (memo-tokenize text)
        reference-tokens (memo-tokenize reference)]
    (pad-left-vec (mapv (fn [a b]
                          (if (= b a)
                            1
                            0))
                        reference-tokens
                        input-tokens)
                  feature-capacity
                  0))
  )

(defn decode
  "Переводит классицикацию перцептрона в значение из памяти.

  result — значение сети.
  memory — ассоциативная память сети"
  [result memory]
  {:pre [(map? memory)]}
  (get memory result)
  )
