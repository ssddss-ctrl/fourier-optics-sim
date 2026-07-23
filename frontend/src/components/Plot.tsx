import Plot from "react-plotly.js";

/**
 * Re-exported so panels import one local module instead of reaching into
 * react-plotly.js directly -- gives us a single place to swap the plotly.js
 * bundle later without touching every panel.
 */
export default Plot;
