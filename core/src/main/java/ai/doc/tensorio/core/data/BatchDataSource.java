/*
 * BatchDataSource.java
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

public interface BatchDataSource {

    /** The batch keys */

    String[] getKeys();

    /** The number of items the data source will vend */

    int size();

    /**
     * Returns the item at index i. It is the responsibility of the data source to randomize
     * (shuffle) the item order
     */

    Batch.Item get(int i);

}
