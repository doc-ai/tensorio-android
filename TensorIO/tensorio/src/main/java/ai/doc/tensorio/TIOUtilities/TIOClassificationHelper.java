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

package ai.doc.tensorio.TIOUtilities;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

public class TIOClassificationHelper {

    /**
     * Orders top N results of a classification
     * @param map Map of labeled floating point classification outputs
     * @param N The number of values to keep
     * @return A list of the top N key-values beginning with the highest one
     */

    public static List<Map.Entry<String, Float>> topN(Map<String,Float> map, int N) {
        PriorityQueue<Map.Entry<String, Float>> queue = topNqueued(map, N);
        List<Map.Entry<String, Float>> list = new ArrayList<Map.Entry<String, Float>>(5);

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

    public static PriorityQueue<Map.Entry<String, Float>> topNqueued(Map<String,Float> map, int N) {
        PriorityQueue<Map.Entry<String, Float>> queue = new PriorityQueue<>(N, (o1, o2) -> (o1.getValue()).compareTo(o2.getValue()));

        for (Map.Entry<String,Float> entry : map.entrySet()) {
            queue.add(new AbstractMap.SimpleEntry<>(entry.getKey(), entry.getValue()));
            if (queue.size() > N) {
                queue.poll();
            }
        }

        return queue;
    }
}
