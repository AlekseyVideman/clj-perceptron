(ns training.math)

(defn dot-product [vector-a vector-b]
  (reduce + (map * vector-a vector-b)))

(defn sign
  [x]
  (if (< x 0)
    -1
    1)
  )
