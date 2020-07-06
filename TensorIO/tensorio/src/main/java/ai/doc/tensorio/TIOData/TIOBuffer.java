/*
 * TIOBuffer.java
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

package ai.doc.tensorio.TIOData;

import java.nio.ByteBuffer;

import ai.doc.tensorio.TIOLayerInterface.TIOLayerDescription;

// TODO: Rename this class and the classes that extend it, correspond to TIOData in obj-c implementation

/**
 * A TIOBuffer is responsible for converting user land data into the format expected by a model in a
 * backend and for converting data from the model back into a user land format.
 *
 * Currently hard coded to a single backend implementation, TF Lite
 */

public abstract class TIOBuffer {

    /**
     * Concrete implementations should override to perform custom initialization
     * @param description A description of the layer this buffer will be used for. Do not hold onto
     *                    the description or you will create a retain loop
     */

    public TIOBuffer(TIOLayerDescription description) {}

    public abstract ByteBuffer toByteBuffer(Object o);
    public abstract Object fromByteBuffer(ByteBuffer buffer);
    public abstract ByteBuffer getBackingByteBuffer();
}
