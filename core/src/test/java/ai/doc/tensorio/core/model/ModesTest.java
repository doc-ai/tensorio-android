/*
 * ModelModesTest.java
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

import org.json.JSONArray;
import org.json.JSONException;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

public class ModesTest {

    @Before
    public void setUp() throws Exception {
    }

    @After
    public void tearDown() throws Exception {
    }

    @Test
    public void testNoModesDefaultsToPredict() {
        try {
            Modes modes1 = new Modes();
            Modes modes2 = new Modes(null);
            Modes modes3 = new Modes(new JSONArray());

            assertTrue(modes1.predicts());
            assertTrue(modes2.predicts());
            assertTrue(modes3.predicts());
        } catch (JSONException e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void testParsesOnlyPredict() {
        JSONArray predicts = new JSONArray();
        predicts.put("predict");

        try {
            Modes modes = new Modes(predicts);
            assertTrue(modes.predicts());
            assertFalse(modes.trains());
            assertFalse(modes.evals());
        } catch (JSONException e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void testParsesOnlyTraining() {
        JSONArray trains = new JSONArray();
        trains.put("train");

        try {
            Modes modes = new Modes(trains);
            assertTrue(modes.trains());
            assertFalse(modes.predicts());
            assertFalse(modes.evals());
        } catch (JSONException e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void testParsesOnlyEval() {
        JSONArray evals = new JSONArray();
        evals.put("eval");

        try {
            Modes modes = new Modes(evals);
            assertTrue(modes.evals());
            assertFalse(modes.predicts());
            assertFalse(modes.trains());
        } catch (JSONException e) {
            e.printStackTrace();
            fail();
        }
    }

    @Test
    public void testParsesMultipleModes() {
        JSONArray multiple = new JSONArray();
        multiple.put("predict");
        multiple.put("train");

        try {
            Modes modes = new Modes(multiple);
            assertTrue(modes.predicts());
            assertTrue(modes.trains());
            assertFalse(modes.evals());
        } catch (JSONException e) {
            e.printStackTrace();
            fail();
        }
    }
}