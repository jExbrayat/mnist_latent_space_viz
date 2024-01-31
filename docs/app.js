document.addEventListener('DOMContentLoaded', function () {
    // Load data from .npy file
    d3.json('your_data.json').then(function (data) {
        // Assuming your data is an array of 100x100 values
        createScatterPlot(data);
    });
});

function createScatterPlot(data) {
    // Set up SVG dimensions
    const width = 500;
    const height = 500;

    // Create an SVG container
    const svg = d3.select('#scatter-plot-container')
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    // Create scales
    const xScale = d3.scaleLinear()
        .domain([0, 100])
        .range([0, width]);

    const yScale = d3.scaleLinear()
        .domain([0, 100])
        .range([height, 0]);

    // Create circles for each data point
    svg.selectAll('circle')
        .data(data)
        .enter()
        .append('circle')
        .attr('cx', (d, i) => xScale(i % 100))  // x position based on column index
        .attr('cy', (d, i) => yScale(Math.floor(i / 100)))  // y position based on row index
        .attr('r', 2)  // radius of the circle
        .attr('fill', (d) => getColorBasedOnValue(d));  // fill color based on data value
}

function getColorBasedOnValue(value) {
    // Add your logic to determine color based on the data value
    // For example, you can use a color scale or a conditional statement
    return 'blue';
}
