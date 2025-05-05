/**
 * Visualization JavaScript for AMR Mutation Analyzer
 * 
 * This file contains functions for creating data visualizations using Chart.js.
 */

/**
 * Create a frequency comparison chart for mutations
 * @param {string} canvasId - The ID of the canvas element
 * @param {object} data - The data for the chart
 */
function createFrequencyComparisonChart(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    
    // Format labels to prevent overflow
    const formattedLabels = data.labels.map(label => {
        if (label.length > 15) {
            return label.substring(0, 12) + '...';
        }
        return label;
    });
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: formattedLabels,
            datasets: [
                {
                    label: 'Resistant Strains',
                    data: data.resistant_frequencies.map(val => val * 100),
                    backgroundColor: 'rgba(220, 53, 69, 0.7)', // Bootstrap danger color
                    borderColor: 'rgba(220, 53, 69, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Non-Resistant Strains',
                    data: data.non_resistant_frequencies.map(val => val * 100),
                    backgroundColor: 'rgba(25, 135, 84, 0.7)', // Bootstrap success color
                    borderColor: 'rgba(25, 135, 84, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + context.raw.toFixed(2) + '%';
                        }
                    }
                },
                legend: {
                    position: 'top',
                },
                title: {
                    display: false,
                    text: 'Mutation Frequency Comparison'
                }
            },
            scales: {
                x: {
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Frequency (%)'
                    },
                    max: 100
                }
            }
        }
    });
}

/**
 * Create a ROC curve chart
 * @param {string} canvasId - The ID of the canvas element
 * @param {object} data - The data for the chart
 */
function createROCChart(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    
    // Add reference line (random classifier)
    const refLineData = [
        { x: 0, y: 0 },
        { x: 1, y: 1 }
    ];
    
    // Create points from the FPR and TPR arrays
    const rocPoints = data.fpr.map((fpr, index) => {
        return { x: fpr, y: data.tpr[index] };
    });
    
    new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'ROC Curve (AUC = ' + data.auc.toFixed(3) + ')',
                    data: rocPoints,
                    showLine: true,
                    backgroundColor: 'rgba(13, 110, 253, 0.7)', // Bootstrap primary color
                    borderColor: 'rgba(13, 110, 253, 1)',
                    pointRadius: 0,
                    borderWidth: 2,
                    fill: false
                },
                {
                    label: 'Random Classifier (AUC = 0.5)',
                    data: refLineData,
                    showLine: true,
                    backgroundColor: 'rgba(173, 181, 189, 0.5)', // Bootstrap secondary color
                    borderColor: 'rgba(173, 181, 189, 0.8)',
                    pointRadius: 0,
                    borderWidth: 1,
                    borderDash: [5, 5],
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const point = context.raw;
                            return 'FPR: ' + point.x.toFixed(3) + ', TPR: ' + point.y.toFixed(3);
                        }
                    }
                },
                legend: {
                    position: 'top',
                },
                title: {
                    display: false,
                    text: 'ROC Curve'
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    beginAtZero: true,
                    max: 1,
                    title: {
                        display: true,
                        text: 'False Positive Rate'
                    }
                },
                y: {
                    beginAtZero: true,
                    max: 1,
                    title: {
                        display: true,
                        text: 'True Positive Rate'
                    }
                }
            }
        }
    });
}

/**
 * Create a feature importance chart
 * @param {string} canvasId - The ID of the canvas element
 * @param {object} data - The data for the chart
 */
function createFeatureImportanceChart(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    
    // Sort data by importance values (descending)
    const sortedIndices = Array.from(Array(data.values.length).keys())
        .sort((a, b) => data.values[b] - data.values[a]);
    
    const sortedLabels = sortedIndices.map(i => {
        const label = data.labels[i];
        return label.length > 15 ? label.substring(0, 12) + '...' : label;
    });
    
    const sortedValues = sortedIndices.map(i => data.values[i]);
    
    // Get colors based on importance (darker = more important)
    const colors = sortedValues.map((val, idx) => {
        const alpha = Math.max(0.4, Math.min(0.9, 0.4 + (val / Math.max(...sortedValues)) * 0.5));
        return `rgba(13, 110, 253, ${alpha})`; // Bootstrap primary color with varying alpha
    });
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: sortedLabels,
            datasets: [
                {
                    label: 'Feature Importance',
                    data: sortedValues,
                    backgroundColor: colors,
                    borderColor: 'rgba(13, 110, 253, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return 'Importance: ' + context.raw.toFixed(4);
                        }
                    }
                },
                legend: {
                    display: false
                },
                title: {
                    display: false,
                    text: 'Feature Importance'
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Importance Score'
                    }
                }
            }
        }
    });
}

/**
 * Create a mutation distribution chart by gene
 * @param {string} canvasId - The ID of the canvas element
 * @param {object} data - The data for the chart
 */
function createDistributionChart(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    
    // Format labels to prevent overflow
    const formattedLabels = data.labels.map(label => {
        if (label.length > 15) {
            return label.substring(0, 12) + '...';
        }
        return label;
    });
    
    // Generate colors for each segment
    const colors = data.values.map((_, idx) => {
        const hue = 200 + (idx * 25) % 160; // Generate different hues
        return `hsla(${hue}, 70%, 60%, 0.8)`;
    });
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: formattedLabels,
            datasets: [
                {
                    data: data.values,
                    backgroundColor: colors,
                    borderColor: 'rgba(33, 37, 41, 0.8)', // Bootstrap dark color
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.raw;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((value / total) * 100).toFixed(1);
                            return `${label}: ${value} (${percentage}%)`;
                        }
                    }
                },
                legend: {
                    position: 'right',
                    labels: {
                        boxWidth: 15,
                        padding: 10
                    }
                },
                title: {
                    display: false,
                    text: 'Mutation Distribution by Gene'
                }
            }
        }
    });
}

/**
 * Create a heatmap visualization for mutation patterns
 * @param {string} canvasId - The ID of the canvas element
 * @param {object} data - The data for the chart
 */
function createMutationHeatmap(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    
    // Prepare data for heatmap
    const chartData = {
        labels: data.sample_labels,
        datasets: data.mutations.map((mutation, i) => {
            return {
                label: mutation.label,
                data: mutation.values,
                backgroundColor: function(context) {
                    const value = context.dataset.data[context.dataIndex];
                    // Color based on value: red for 1 (mutation present), blue for 0 (no mutation)
                    return value === 1 ? 'rgba(220, 53, 69, 0.8)' : 'rgba(13, 110, 253, 0.2)';
                }
            };
        })
    };
    
    // Create the heatmap
    new Chart(ctx, {
        type: 'matrix',
        data: chartData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        title: function(context) {
                            return `Sample: ${context[0].label}`;
                        },
                        label: function(context) {
                            const value = context.dataset.data[context.dataIndex];
                            return `${context.dataset.label}: ${value === 1 ? 'Present' : 'Absent'}`;
                        }
                    }
                },
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Mutation Patterns Across Samples'
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Samples'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Mutations'
                    }
                }
            }
        }
    });
}
