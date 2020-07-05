package ai.doc.tensorio.TIOModel;

import org.json.JSONArray;
import org.json.JSONException;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

public class TIOModelModesTest {

    @Before
    public void setUp() throws Exception {
    }

    @After
    public void tearDown() throws Exception {
    }

    @Test
    public void testNoModesDefaultsToPredict() {
        try {
            TIOModelModes modes1 = new TIOModelModes();
            TIOModelModes modes2 = new TIOModelModes(null);
            TIOModelModes modes3 = new TIOModelModes(new JSONArray());

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
            TIOModelModes modes = new TIOModelModes(predicts);
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
            TIOModelModes modes = new TIOModelModes(trains);
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
            TIOModelModes modes = new TIOModelModes(evals);
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
            TIOModelModes modes = new TIOModelModes(multiple);
            assertTrue(modes.predicts());
            assertTrue(modes.trains());
            assertFalse(modes.evals());
        } catch (JSONException e) {
            e.printStackTrace();
            fail();
        }
    }
}