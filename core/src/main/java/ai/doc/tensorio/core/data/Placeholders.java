package ai.doc.tensorio.core.data;

import java.util.HashMap;

/**
 * Placeholders are named layers in a model that can have values set before running the model
 * but which are not explicitly marked as inputs to the model. Placeholders are often used for
 * hyperparameters. From a data structure perspective they are no different than model inputs
 * but are explicitly typed for clarity and passed as a separate paramater to the model.
 */

public class Placeholders extends HashMap<String, Object> {

}
