/*
 * TIOModelIO.java
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

package ai.doc.tensorio.TIOModel;

import android.support.annotation.Nullable;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.List;
import java.util.Set;

import ai.doc.tensorio.TIOLayerInterface.TIOLayerInterface;

/**
 * Encapsulates information about the inputs, outputs, and placeholders for a model.
 */

public class TIOModelIO {

    /**
     * An I/O list may be indexed by key or by index.
     */

    public class TIOModelIOList {

        private List<TIOLayerInterface> indexedInterfaces;
        private Map<String, TIOLayerInterface> namedInterfaces;
        private Map<String, Integer> nameToIndex;

        /**
         * Initializes an indexed model list with a list interfaces. You should not
         * need to create instances of this class yourself.
         *
         * If the initializing interfaces parameter is nil, it will be treated as an
         * empty list.
         */

        public TIOModelIOList(@Nullable List<TIOLayerInterface> interfaces) {
            this.namedInterfaces = new HashMap<>();
            this.nameToIndex = new HashMap<>();

            if ( interfaces == null) {
                this.indexedInterfaces = new ArrayList<>();
            } else {
                this.indexedInterfaces = interfaces;

                for (int i = 0; i < interfaces.size(); i++) {
                    TIOLayerInterface layerInterface = interfaces.get(i);
                    this.namedInterfaces.put(layerInterface.getName(), layerInterface);
                    this.nameToIndex.put(layerInterface.getName(), Integer.valueOf(i));
                }
            }
        }

        /**
         * The number of items in the list.
         */

        public int size() {
            return this.indexedInterfaces.size();
        }

        /**
         * Returns the `TIOLayerInterface` for the I/O at a numeric index, or raises an
         * exception if no interface is available at the index.
         */

        public TIOLayerInterface get(Integer index) {
            return this.indexedInterfaces.get(index);
        }

        /**
         * Returns the `TIOLayerInterface` for the I/O at a named index, or raises an
         * exception if no interface is available at the index.
         */

        public TIOLayerInterface get(String name) {
            return this.namedInterfaces.get(name);
        }

        /**
         * All items in the list as a List.
         */

        public List<TIOLayerInterface> all() {
            return this.indexedInterfaces;
        }

        /**
         * All of the keys (names) in the list.
         */

        public Set<String> keys() {
            return this.namedInterfaces.keySet();
        }

        /**
         * Returns the numeric index for the named index.
         */

        public Integer indexFor(String name) {
            return this.nameToIndex.get(name);
        }
    }

    private TIOModelIOList inputs;
    private TIOModelIOList outputs;
    private TIOModelIOList placeholders;

    /**
     * Initializes an instance of TIOModelIO with input and output interfaces.
     */

    public TIOModelIO(List<TIOLayerInterface> inputInterfaces, List<TIOLayerInterface> outputInterfaces) {
        this.inputs = new TIOModelIOList(inputInterfaces);
        this.outputs = new TIOModelIOList(outputInterfaces);
    }

    /**
     * Initializes an instance of TIOModelIO with input, output, and placeholder interfaces.
     */

    public TIOModelIO(List<TIOLayerInterface> inputInterfaces, List<TIOLayerInterface> outputInterfaces, List<TIOLayerInterface> placeholderInterfaces) {
        this(inputInterfaces, outputInterfaces);
        this.placeholders = new TIOModelIOList(placeholderInterfaces);
    }

    /**
     * The inputs list. Access the values in this list using indexed subscripting
     * by name or by key.
     *
     * @code
     * getInputs(0)
     * getInputs("image")
     * @endcode
     */

    public TIOModelIOList getInputs() {
        return inputs;
    }

    /**
     * The outputs list. Access the values in this list using indexed subscripting
     * by name or by key.
     *
     * @code
     * getOutputs(0)
     * getOutputs("label")
     * @endcode
     */

    public TIOModelIOList getOutputs() {
        return outputs;
    }

    /**
     * The placeholders list. May be empty. Access the values in this list using
     * indexed subscripting by name or by key.
     *
     * @code
     * getPlaceholders(0)
     * getPlaceholders("label")
     * @endcode
     */

    public TIOModelIOList getPlaceholders() {
        return placeholders;
    }
}
