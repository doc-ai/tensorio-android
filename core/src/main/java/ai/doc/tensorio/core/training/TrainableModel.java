/*
 * TrainableModel.java
 * TensorIO
 *
 * Created by Philip Dow on 10/20/2020
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

package ai.doc.tensorio.core.training;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.io.File;
import java.util.Map;

import ai.doc.tensorio.core.data.Batch;
import ai.doc.tensorio.core.data.Placeholders;
import ai.doc.tensorio.core.model.Model;

public interface TrainableModel {

    /**
     * Perform training on an map of objects
     * @param inputs A mapping of layer names to arbitrary objects
     * @return results of running the model mapped from the output layer names to the values
     * @throws Model.ModelException Raised if the model has not yet been loaded and the attempt to
     *                           load it fails
     * @throws IllegalArgumentException Raised if the input to the model does not conform to the
     *                                  expected inputs
     */

    Map<String, Object> trainOn(@NonNull Map<String, Object> inputs) throws Model.ModelException, IllegalArgumentException;

    /**
     * Perform training on a map of of objects with placeholders. Not all backends support placeholders,
     * in which case concrete implementations should raise an exception.
     *
     * @param inputs A batch of items mapping layer names to arbitrary objects
     * @param placeholders A mapping of placeholder layer names to arbitrary objects. May be nil, in
     *                     which case calling this method should be no different from calling runOn
     *                     without placeholders.
     * @return results of running the model mapped from the output layer names to the values
     * @throws Model.ModelException Raised if the model has not yet been loaded and the attempt to
     *                           load it fails
     * @throws IllegalArgumentException Raised if the input to the model does not conform to the
     *                                  expected inputs
     */

    Map<String, Object> trainOn(@NonNull Map<String, Object> inputs, @Nullable Placeholders placeholders) throws Model.ModelException, IllegalArgumentException;

    /**
     * Perform training on a batch of objects
     * @param batch A batch of items mapping layer names to arbitrary objects
     * @return results of running the model mapped from the output layer names to the values
     * @throws Model.ModelException Raised if the model has not yet been loaded and the attempt to
     *                           load it fails
     * @throws IllegalArgumentException Raised if the input to the model does not conform to the
     *                                  expected inputs
     */

    Map<String, Object> trainOn(@NonNull Batch batch) throws Model.ModelException, IllegalArgumentException;

    /**
     * Perform training on a batch of objects with placeholders. Not all backends support placeholders,
     * in which case concrete implementations should raise an exception.
     *
     * @param batch A batch of items mapping layer names to arbitrary objects
     * @param placeholders A mapping of placeholder layer names to arbitrary objects. May be nil, in
     *                     which case calling this method should be no different from calling runOn
     *                     without placeholders.
     * @return results of running the model mapped from the output layer names to the values
     * @throws Model.ModelException Raised if the model has not yet been loaded and the attempt to
     *                           load it fails
     * @throws IllegalArgumentException Raised if the input to the model does not conform to the
     *                                  expected inputs
     */

    Map<String, Object> trainOn(@NonNull Batch batch, Placeholders placeholders) throws Model.ModelException, IllegalArgumentException;

    /**
     * Exports the model checkpoints to a directory, used to write updated checkpoints to disk after training
     */

    void exportTo(File file);
}
