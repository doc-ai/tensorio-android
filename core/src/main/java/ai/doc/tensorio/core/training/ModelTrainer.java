/*
 * ModelTrainer.java
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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import ai.doc.tensorio.core.data.Batch;
import ai.doc.tensorio.core.data.BatchDataSource;
import ai.doc.tensorio.core.data.Placeholders;
import ai.doc.tensorio.core.model.Model.ModelException;

public class ModelTrainer {

    /** The model to train */

    private final @NonNull TrainableModel model;

    /** The data source that will vend batches of data for training */

    private final @NonNull BatchDataSource dataSource;

    /** The placeholders that will be set before training */

    private final @Nullable Placeholders placeholders;

    /** The number of epochs to train for */

    private final int epochs;

    /** The number of items in a batch */

    private final int batchSize;

    /** Whether data should be requested from the data source in a random order */

    private final boolean shuffle;

    /** Default constructor */

    public ModelTrainer(@NonNull TrainableModel model, @NonNull BatchDataSource dataSource, @Nullable Placeholders placeholders, int epochs, int batchSize, boolean shuffle) {
        this.model = model;
        this.dataSource = dataSource;
        this.placeholders = placeholders;
        this.epochs = epochs;
        this.batchSize = batchSize;
        this.shuffle = shuffle;
    }

    /**
     * Trains the model using the parameters provided at instantiation, one batch at a time
     * from the data source. Returns the results acquired on the last training set, typically
     * the loss value.
     *
     * @return The final results of training
     * @throws ModelException If there is a problem training the model
     */

    public Map<String, Object> train() throws ModelException {
        Map<String, Object> results = null;
        int batchCount = batchCount();
        prepareItemOrder();

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int index = 0; index < batchCount; index++) {
                results = model.trainOn(batch(index), placeholders);
            }
        }

        return results;
    }

    /**
     * Trains the model using the parameters provided at instantiation, one batch at a time from
     * the data source. Calls the callback at the end of each epoch with the most recently acquired
     * training results, typically the loss value
     *
     * @param callback The function to call at the end of each training object
     * @throws ModelException If there is a problem training the model
     */

    public void train(Consumer<Map<String, Object>> callback) throws ModelException {
        Map<String, Object> results = null;
        int batchCount = batchCount();
        prepareItemOrder();

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int index = 0; index < batchCount; index++) {
                results = model.trainOn(batch(index), placeholders);
            }
            callback.accept(results);
        }
    }

    /** Returns the number of batches given the number of items vended by the data source and the batch size */

    private int batchCount() {
        return (int) Math.ceil( (double)dataSource.size() / (double)batchSize );
    }

    /** Returns the batch at index i */

    private Batch batch(int index) {
        Batch batch = new Batch(dataSource.getKeys());
        int size = dataSource.size();

        int start = index * batchSize;
        int count = 0;

        if ( start + batchSize > size ) {
            count = size - start;
        } else {
            count = batchSize;
        }

        for (int i = start; i < start + count; i++) {
            batch.add(dataSource.get(itemOrder.get(i)));
        }

        return batch;
    }

    /** (Randomized) item order */

    private List<Integer> itemOrder;

    /** Randomizes the order of the items that will be requested from the data source */

    private void prepareItemOrder() {
        List<Integer> values = new ArrayList<>();
        for (int i = 0; i < dataSource.size(); i++ ) {
            values.add(i);
        }
        if (shuffle) {
            Collections.shuffle(values);
        }
        itemOrder = values;
    }
}
