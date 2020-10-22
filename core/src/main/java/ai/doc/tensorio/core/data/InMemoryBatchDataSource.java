/*
 * InMemoryBatchDataSource.java
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

package ai.doc.tensorio.core.data;

public class InMemoryBatchDataSource implements BatchDataSource {

    /** The batch of items that will be vended by this data source */

    private final Batch batch;

    /** Instantiates the data source with a batch */

    public InMemoryBatchDataSource(Batch batch) {
        this.batch = batch;
    }

    /** Instantiates this data source with a single item */

    public InMemoryBatchDataSource(Batch.Item item) {
        this.batch = new Batch(item);
    }

    // Batch Data Source

    public String[] getKeys() {
        return batch.getKeys();
    }

    public int size() {
        return batch.size();
    }

    public Batch.Item get(int i) {
        return batch.get(i);
    }
}
