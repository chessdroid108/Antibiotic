// Main JavaScript for AMR Mutation Analyzer

document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    })

    // Handle file input for genome upload
    const fileInput = document.getElementById('file-input');
    const fileLabel = document.getElementById('file-label');

    if (fileInput && fileLabel) {
        fileInput.addEventListener('change', function() {
            if (fileInput.files.length > 0) {
                fileLabel.textContent = fileInput.files[0].name;
            } else {
                fileLabel.textContent = 'Choose file';
            }
        });
    }

    // Add event listeners for resistance radio buttons
    const resistantYes = document.getElementById('resistant-yes');
    const resistantNo = document.getElementById('resistant-no');
    const resistantUnknown = document.getElementById('resistant-unknown');
    const antibioticField = document.getElementById('antibiotic-field');

    if (resistantYes && resistantNo && resistantUnknown && antibioticField) {
        function toggleAntibioticField() {
            if (resistantYes.checked || resistantNo.checked) {
                antibioticField.style.display = 'block';
            } else {
                antibioticField.style.display = 'none';
            }
        }

        resistantYes.addEventListener('change', toggleAntibioticField);
        resistantNo.addEventListener('change', toggleAntibioticField);
        resistantUnknown.addEventListener('change', toggleAntibioticField);

        // Initialize on page load
        toggleAntibioticField();
    }

    // Initialize any charts if present on the page
    initializeCharts();
});

// Function to initialize charts on results page
function initializeCharts() {
    // ROC Curve Chart
    const rocCanvas = document.getElementById('roc-curve-chart');
    if (rocCanvas) {
        const rocData = JSON.parse(rocCanvas.dataset.chartData);
        new Chart(rocCanvas, {
            type: 'line',
            data: {
                labels: rocData.fpr,
                datasets: [{
                    label: 'ROC Curve',
                    data: rocData.tpr,
                    borderColor: '#0d6efd',
                    backgroundColor: 'rgba(13, 110, 253, 0.1)',
                    fill: true,
                    tension: 0.4
                }, {
                    label: 'Random Classifier',
                    data: rocData.fpr,
                    borderColor: '#6c757d',
                    borderDash: [5, 5],
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'False Positive Rate'
                        },
                        beginAtZero: true,
                        max: 1
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'True Positive Rate'
                        },
                        beginAtZero: true,
                        max: 1
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'ROC Curve (AUC: ' + rocData.auc.toFixed(3) + ')'
                    }
                }
            }
        });
    }

    // Mutation Frequency Chart
    const freqCanvas = document.getElementById('mutation-frequency-chart');
    if (freqCanvas) {
        const freqData = JSON.parse(freqCanvas.dataset.chartData);
        new Chart(freqCanvas, {
            type: 'bar',
            data: {
                labels: freqData.mutations,
                datasets: [
                    {
                        label: 'Resistant Strains',
                        data: freqData.resistant_freq,
                        backgroundColor: 'rgba(220, 53, 69, 0.7)',
                        borderColor: 'rgba(220, 53, 69, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Non-Resistant Strains',
                        data: freqData.non_resistant_freq,
                        backgroundColor: 'rgba(25, 135, 84, 0.7)',
                        borderColor: 'rgba(25, 135, 84, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Mutation'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Frequency'
                        },
                        min: 0,
                        max: 1
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Mutation Frequencies in Resistant vs. Non-Resistant Strains'
                    }
                }
            }
        });
    }

    // Confusion Matrix Heatmap
    const cmCanvas = document.getElementById('confusion-matrix-chart');
    if (cmCanvas) {
        const cmData = JSON.parse(cmCanvas.dataset.chartData);
        new Chart(cmCanvas, {
            type: 'matrix',
            data: {
                datasets: [{
                    label: 'Confusion Matrix',
                    data: [
                        { x: 'Actual Negative', y: 'Predicted Negative', v: cmData.tn },
                        { x: 'Actual Negative', y: 'Predicted Positive', v: cmData.fp },
                        { x: 'Actual Positive', y: 'Predicted Negative', v: cmData.fn },
                        { x: 'Actual Positive', y: 'Predicted Positive', v: cmData.tp }
                    ],
                    backgroundColor(ctx) {
                        const value = ctx.dataset.data[ctx.dataIndex].v;
                        const maxValue = Math.max(cmData.tn, cmData.fp, cmData.fn, cmData.tp);
                        const alpha = value / maxValue;
                        return `rgba(13, 110, 253, ${alpha})`;
                    },
                    borderColor: 'white',
                    borderWidth: 1,
                    width: ({ chart }) => (chart.chartArea || {}).width / 2 - 1,
                    height: ({ chart }) => (chart.chartArea || {}).height / 2 - 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    tooltip: {
                        callbacks: {
                            title() {
                                return '';
                            },
                            label(context) {
                                const v = context.dataset.data[context.dataIndex];
                                return [`${v.y} / ${v.x}: ${v.v}`];
                            }
                        }
                    },
                    title: {
                        display: true,
                        text: 'Confusion Matrix'
                    },
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
}
