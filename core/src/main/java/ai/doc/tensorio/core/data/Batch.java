/*
 * Batch.java
 * TensorIO
 *
 * Created by Philip Dow on 10/29/2020
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

package ai.doc.tensorio.core.data;

import androidx.annotation.NonNull;

import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

public final class Batch extends AbstractList<Batch.Item> {

    /** A batch item is just a map of keys to values */

    public static class Item extends HashMap<String, Object> { }

    /** The internal representation of items */

    // Internally items are managed as a collection of named arrays of data. Think of the collection
    // as a matrix whose rows are any single item and whose columns are named. Values are accessed
    // by row index (get) or by column name (valuesForKey).

    private Map<String, List<Object>> items;

    // Private Properties

    /** The number of items in the batch */

    private int count = 0;

    /** The batch keys */

    private String[] keys;

    /** The batch keys cached as a Set */

    private Set<String> keyset;

    // Constructors

    /** Instantiates a batch with keys. You must add items with those keys to it */

    public Batch(@NonNull String[] keys) {
        this.keys = keys;
        this.keyset = new HashSet<String>(Arrays.asList(this.keys));
    }

    /** Instantiates a batch with items. The items must have the same keys */

    public Batch(@NonNull Item[] items) throws IllegalArgumentException {
        Set<String> keys = items[0].keySet();
        this.keys = (String[]) keys.toArray(new String[0]);
        this.keyset = new HashSet<String>(Arrays.asList(this.keys));
        for (Item item : items) {
            add(item);
        }
    }

    /** Instantiates a batch with a single item */

    public Batch(@NonNull Item item) {
        Set<String> keys = item.keySet();
        this.keys = (String[]) keys.toArray(new String[0]);
        this.keyset = new HashSet<String>(Arrays.asList(this.keys));
        add(item);
    }

    // Public Getters and Setters

    /** Returns the item count, use size() */

    public int getCount() {
        return count;
    }

    /** Returns the batch keys */

    public String[] getKeys() {
        return keys;
    }

    /** Returns the keyset */

    public Set<String> getKeyset() {
        return keyset;
    }

    // List Implementation

    /** Returns the number of items in the batch */

    @Override
    public int size() {
        return count;
    }

    /** Retrieves the item at index i in the batch */

    @Override
    public Item get(int i) {
        Batch.Item item  = new Batch.Item();

        for (String key : keys) {
            item.put(key, Objects.requireNonNull(items.get(key)).get(i));
        }

        return item;
    }

    // Additional Methods

    /** Adds an item to the batch. It must have the same keys that the batch was instantiated with */

    public boolean add(Item item) throws IllegalArgumentException {
        validateKeys(item);
        createItemsMap();
        count++;

        for (Map.Entry<String, Object> entry : item.entrySet()) {
            Objects.requireNonNull(items.get(entry.getKey())).add(entry.getValue());
        }

        return true;
    }

    /** Returns all the value for a particular key, effectively a column of data */

    public Object[] valuesForKey(String key) {
        return Objects.requireNonNull(items.get(key)).toArray(new Object[0]);
    }

    /** Convenience method that calls valuesForKey */

    public Object[] get(String name) {
        return valuesForKey(name);
    }

    /** Instantiates the item map */

    private void createItemsMap() {
        if (items != null) {
            return;
        }

        items = new HashMap<>();

        for (String key : keys) {
            items.put(key, new ArrayList<Object>());
        }
    }

    /** Validates that the item's keys match the keys contained in the batch */

    private void validateKeys(Item item) throws IllegalArgumentException {
        if (!keyset.equals(item.keySet())) {
            throw new IllegalArgumentException("Keys in item do not match batch keys");
        }
    }
}
