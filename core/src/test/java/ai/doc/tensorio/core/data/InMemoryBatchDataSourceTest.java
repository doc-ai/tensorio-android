/*
 * InMemoryBatchDataSourceTest.java
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

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.*;

public class InMemoryBatchDataSourceTest {

    private Batch.Item item1;
    private Batch.Item item2;

    @Before
    public void setUp() throws Exception {
        createItem1();
        createItem2();
    }

    @After
    public void tearDown() throws Exception {
        item1 = null;
        item2 = null;
    }

    private void createItem1() {
        float[] input = {
                1,2,3,4
        };
        float[] label = {
                0
        };

        item1 = new Batch.Item();
        item1.put("input", input);
        item1.put("label", label);
    }

    private void createItem2() {
        float[] input = {
                5,6,7,8
        };
        float[] label = {
                1
        };

        item2 = new Batch.Item();
        item2.put("input", input);
        item2.put("label", label);
    }

    @Test
    public void testTakesKeysFromBatch() {
        Batch.Item[] items = {item1, item2};
        Batch batch = new Batch(items);
        InMemoryBatchDataSource dataSource = new InMemoryBatchDataSource(batch);

        assertEquals(2, dataSource.getKeys().length);
        assertTrue(Arrays.asList(dataSource.getKeys()).contains("input"));
        assertTrue(Arrays.asList(dataSource.getKeys()).contains("label"));
    }

    @Test
    public void testTakesKeysFromItem() {
        InMemoryBatchDataSource dataSource = new InMemoryBatchDataSource(item1);

        assertEquals(2, dataSource.getKeys().length);
        assertTrue(Arrays.asList(dataSource.getKeys()).contains("input"));
        assertTrue(Arrays.asList(dataSource.getKeys()).contains("label"));
    }

    @Test
    public void testGetsSizeFromBatch() {
        Batch.Item[] items = {item1, item2};
        Batch batch = new Batch(items);
        InMemoryBatchDataSource dataSource = new InMemoryBatchDataSource(batch);
        assertEquals(2, dataSource.size());
    }

    @Test
    public void testGetsSizeFromItem() {
        InMemoryBatchDataSource dataSource = new InMemoryBatchDataSource(item1);
        assertEquals(1, dataSource.size());
    }

    @Test
    public void testReturnsItemsFromBatch() {
        Batch.Item[] items = {item1, item2};
        Batch batch = new Batch(items);
        InMemoryBatchDataSource dataSource = new InMemoryBatchDataSource(batch);
        Batch.Item itemA = dataSource.get(0);
        Batch.Item itemB = dataSource.get(1);
        assertEquals(item1, itemA);
        assertEquals(item2, itemB);
    }

    @Test
    public void testReturnsItemFromItem() {
        InMemoryBatchDataSource dataSource = new InMemoryBatchDataSource(item1);
        Batch.Item item = dataSource.get(0);
        assertEquals(item1, item);
    }

}