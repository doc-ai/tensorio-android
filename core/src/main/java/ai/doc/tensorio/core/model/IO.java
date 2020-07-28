/*
 * ModelIO.java
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

package ai.doc.tensorio.core.model;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.List;
import java.util.Set;

import ai.doc.tensorio.core.layerinterface.LayerInterface;

/**
 * Encapsulates information about the inputs, outputs, and placeholders for a model.
 */

public class IO {

    /**
     * An I/O list may be indexed by key or by index.
     */

    public class IOList {

        private List<LayerInterface> indexedInterfaces;
        private Map<String, LayerInterface> namedInterfaces;
        private Map<String, Integer> nameToIndex;

        /**
         * Initializes an indexed model list with a list interfaces. You should not
         * need to create instances of this class yourself.
         *
         * If the initializing interfaces parameter is nil, it will be treated as an
         * empty list.
         */

        public IOList(@Nullable List<LayerInterface> interfaces) {
            this.namedInterfaces = new HashMap<>();
            this.nameToIndex = new HashMap<>();

            if ( interfaces == null) {
                this.indexedInterfaces = new ArrayList<>();
            } else {
                this.indexedInterfaces = interfaces;

                for (int i = 0; i < interfaces.size(); i++) {
                    LayerInterface layerInterface = interfaces.get(i);
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
         * Returns the `LayerInterface` for the I/O at a numeric index, or raises an
         * exception if no interface is available at the index.
         */

        public LayerInterface get(Integer index) {
            return this.indexedInterfaces.get(index);
        }

        /**
         * Returns the `LayerInterface` for the I/O at a named index, or raises an
         * exception if no interface is available at the index.
         */

        public LayerInterface get(String name) {
            return this.namedInterfaces.get(name);
        }

        /**
         * All items in the list as a List.
         */

        public List<LayerInterface> all() {
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

    private IOList inputs;
    private IOList outputs;
    private IOList placeholders;

    /**
     * Initializes an instance of IO with input and output interfaces.
     */

    public IO(@NonNull List<LayerInterface> inputInterfaces, @NonNull List<LayerInterface> outputInterfaces) {
        this.inputs = new IOList(inputInterfaces);
        this.outputs = new IOList(outputInterfaces);
    }

    /**
     * Initializes an instance of IO with input, output, and placeholder interfaces.
     */

    public IO(@NonNull List<LayerInterface> inputInterfaces, @NonNull List<LayerInterface> outputInterfaces, @Nullable List<LayerInterface> placeholderInterfaces) {
        this(inputInterfaces, outputInterfaces);
        this.placeholders = new IOList(placeholderInterfaces);
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

    public IOList getInputs() {
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

    public IOList getOutputs() {
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

    public IOList getPlaceholders() {
        return placeholders;
    }
}
