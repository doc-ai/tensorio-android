/*
 * StringConverter.java
 * TensorIO
 *
 * Created by Sam Leroux on 12/15/2020
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


package ai.doc.tensorio.pytorch.data;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import org.pytorch.Tensor;

import java.nio.ByteBuffer;

import ai.doc.tensorio.core.layerinterface.LayerDescription;

public class StringConverter implements Converter {

    // TODO: Implement. Strings are really just byte arrays, it just happens to be the terminology TensorFlow uses for byte arrays

    @Override
    public Tensor toTensor(@NonNull Object o, @NonNull LayerDescription description, @Nullable ByteBuffer cache) throws IllegalArgumentException {
        return null;
    }

    @Override
    public Object fromTensor(@NonNull Tensor t, @NonNull LayerDescription description) {
        return null;
    }

    @Override
    public ByteBuffer createBackingBuffer(LayerDescription stringLayer) {
        return null;
    }
}
