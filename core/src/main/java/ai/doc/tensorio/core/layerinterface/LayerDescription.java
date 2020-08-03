/*
 * LayerDescription.java
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

package ai.doc.tensorio.core.layerinterface;

/**
 * Describes an input or output layer. Used internally by a model when parsing its description.
 *
 * A layer description encapsulates information about an input or output tensor that is needed
 * to prepare obj-c data and copy bytes into and out of it. For example, a vector layer description
 * for an input tensor describes any transformations the submitted data must undergo before the
 * underlying bytes are copied to the tensor, e.g. quantization and normalization, as well as the
 * shape of the expected input, which determines how many bytes are copied into the tensor.
 */

public abstract class LayerDescription {

    /**
     * true if this layer is quantized (bytes of type byte), false if not (bytes of type float)
     */

    protected boolean quantized;

    /**
     * true if this layer supports batched inference and training
     */

    protected boolean batched;

    public boolean isQuantized() {
        return quantized;
    }

    public boolean isBatched() {
        return batched;
    }

}
