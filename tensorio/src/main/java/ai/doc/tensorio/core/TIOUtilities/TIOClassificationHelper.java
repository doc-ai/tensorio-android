/*
 * TIOClassificationHelper.java
 * TensorIO
 *
 * Created by Philip Dow on 7/6/2020
 * Copyright (c) 2020 - Present doc.ai (http://doc.ai)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ai.doc.tensorio.core.TIOUtilities;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.stream.Collectors;

import androidx.annotation.NonNull;

public class TIOClassificationHelper {

    /**
     * Orders top N results of a classification
     * @param map Map of labeled floating point classification outputs
     * @param N The number of values to keep
     * @return A list of the top N key-values beginning with the highest one
     */

    public static List<Map.Entry<String, Float>> topN(@NonNull Map<String,Float> map, int N) {
        return topN(map, N, 0);
    }

    /**
     * Orders top N results of a classification whose values are greater than the threshold
     * @param map Map of labeled floating point classification outputs
     * @param N The number of values to keep
     * @param threshold The minimum value to keep
     * @return A list of the top N key-values beginning with the highest one
     */

    public static List<Map.Entry<String, Float>> topN(@NonNull Map<String,Float> map, int N, float threshold) {
        PriorityQueue<Map.Entry<String, Float>> queue = topNQueued(map, N, threshold);
        List<Map.Entry<String, Float>> list = new ArrayList<>(5);

        while (queue.size() > 0) {
            Map.Entry<String, Float> entry = (Map.Entry<String, Float>) queue.poll();
            list.add(0, entry);
        }

        return list;
    }

    /**
     * Orders top N results of a classification using a PriorityQueue. Note the queue's head begins
     * with the lowest value, not the highest, as you might expect. Use topN to have the results ordered
     * from highest to lowest.
     * @param map Map of labeled floating point classification outputs
     * @param N The number of values to keep
     * @return A PriorityQueue of the top N key-values
     */

    public static PriorityQueue<Map.Entry<String, Float>> topNQueued(@NonNull Map<String,Float> map, int N, float threshold) {
        PriorityQueue<Map.Entry<String, Float>> queue = new PriorityQueue<>(N, (o1, o2) -> (o1.getValue()).compareTo(o2.getValue()));

        for (Map.Entry<String,Float> entry : map.entrySet()) {
            if (entry.getValue() < threshold) {
                continue;
            }
            queue.add(new AbstractMap.SimpleEntry<>(entry.getKey(), entry.getValue()));
            if (queue.size() > N) {
                queue.poll();
            }
        }

        return queue;
    }

    /**
     * Applies an exponential decay function to classification results with a decay rate of 0.8 and
     * a threshold of 0.1. May be applied to any list of String to Float values. So that:
     *
     * <code>
     *     x_out = 0.8 * x_previous + 0.2 * x_new
     *     x_out = 0 if x_out < 0.1
     * </code>
     *
     * @param previousValues The previous classification results, or more likely the results of applying this function
     * @param newValues The current classification results
     * @return A list of entries whose values are greater than the default threshold of 0.1
     */

    public static List<Map.Entry<String, Float>> smoothClassification(@NonNull  List<Map.Entry<String, Float>> previousValues, @NonNull List<Map.Entry<String, Float>> newValues) {
        return smoothClassification(previousValues, newValues, 0.8f, 0.1f);
    }

    /**
     * Applies an exponential decay function to classification results with a given decay rate and
     * threshold. May be applied to any list of String to Float values. So that:
     *
     * <code>
     *     x_out = decay * x_previous + (1-decay) * x_new
     *     x_out = 0 if x_out < threshold
     * </code>
     *
     * @param previousValues The previous classification results, or more likely the results of applying this function
     * @param newValues The current classification results
     * @param decay The exponential decay rate to apply
     * @param threshold The threshold below which result will be removed from the output
     * @return A list of entries whose values are greater than the threshold
     */

    public static List<Map.Entry<String, Float>> smoothClassification(@NonNull List<Map.Entry<String, Float>> previousValues, @NonNull List<Map.Entry<String, Float>> newValues, float decay, float threshold) {
        Map<String, Float> previousMap = previousValues.stream().collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
        Map<String, Float> newMap = newValues.stream().collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
        Map<String, Float> combinedMap = new HashMap<>();
        Map<String, Float> outMap = new HashMap<>();
        float update = 1.0f - decay;

        for (Map.Entry<String, Float> entry : previousMap.entrySet()) {
            combinedMap.put(entry.getKey(), entry.getValue() * decay);
        }

        for (Map.Entry<String, Float> entry : newMap.entrySet()) {
            combinedMap.put(entry.getKey(), combinedMap.getOrDefault(entry.getKey(), 0.0f) + entry.getValue() * update);
        }

        for (Map.Entry<String, Float> entry : combinedMap.entrySet()) {
            if (entry.getValue() >= threshold) {
                outMap.put(entry.getKey(), combinedMap.get(entry.getKey()));
            }
        }

        return new ArrayList<>(outMap.entrySet());
    }
}
