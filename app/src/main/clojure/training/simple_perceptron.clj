(ns training.simple-perceptron
  "Перцептрон с одним скрытым склоем и пороговой передаточной функцией.

  Сеть состоит из одного некйрона, потому что длина сообщения зарнее известна,
      что позволяет создавать вектор признаков длина которого = длине сообщеня.
  Если бы сеть работала с большими текстами,
      я бы создал несколько нейронов и разбивал текст на чанки."
  (:require [training.math :as m]))

(defn activate-neuron
  "Модель искуственного нейрона Мак-Каллока — Питтса.

  features — вектор признаков
  weights — вектор весов
  bias — пороговое значение"
  [features weights bias]
  {
   :pre [(vector? features)
         (vector? weights)
         (= (count features) (count weights))]
   }
  (m/sign (- (m/dot-product features weights) bias))
  )

(defn train
  "Обучает перцептрон. Возвращает настроенные веса и смещения

  features — вектор признаков входных данных
  learn-rate — скорость обучения (шаг корректировки весов)"
  [features weights learn-rate bias epoch epoch-limit]
  {:pre [(vector? features)
         (vector? weights)
         (> learn-rate 0)
         (>= epoch 0)
         (> epoch-limit 0)]
   }
  (println (format "Features %s | Epoch %s\\%s" features epoch epoch-limit))
  (let [guess (activate-neuron features weights bias)
        correct 1
        error (- correct guess)]

    (if (>= epoch epoch-limit)
      {:weights weights, :bias bias}

      (let [weight-adjustments (map #(* error learn-rate %) features)
            new-weights (vec (map + weights weight-adjustments))
            new-bias (+ bias (* learn-rate error))]
        (train features new-weights learn-rate new-bias (inc epoch) epoch-limit))))
  )