// WhatsApp Analyzer - Main Script

document.addEventListener('DOMContentLoaded', function() {
    console.log('WhatsApp Analyzer loaded');
});

// File upload handling
function handleFileUpload(file) {
    console.log('File selected:', file.name);
}

// Chart helper function (if needed)
function createChart(ctx, data) {
    return new Chart(ctx, {
        type: 'doughnut',
        data: data,
        options: {
            responsive: true
        }
    });
}