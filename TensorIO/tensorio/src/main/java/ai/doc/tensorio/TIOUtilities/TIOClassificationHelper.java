package ai.doc.tensorio.TIOUtilities;

import java.util.AbstractMap;
import java.util.Map;
import java.util.PriorityQueue;

public class TIOClassificationHelper {

    public static PriorityQueue<Map.Entry<String, Float>> topN(Map<String,Float> map, int N) {
        PriorityQueue<Map.Entry<String, Float>> sortedLabels = new PriorityQueue<>(N, (o1, o2) -> (o1.getValue()).compareTo(o2.getValue()));

        for (Map.Entry<String,Float> entry : map.entrySet()) {
            sortedLabels.add(new AbstractMap.SimpleEntry<>(entry.getKey(), entry.getValue()));
            if (sortedLabels.size() > N) {
                sortedLabels.poll();
            }
        }

        return sortedLabels;
    }
}
