$(document).ready(function() {
    // Handle file input change
    $('#image-upload').change(function(e) {
        const file = e.target.files[0];
        if (file) {
            // Show preview
            const reader = new FileReader();
            reader.onload = function(e) {
                $('#preview-image')
                    .attr('src', e.target.result)
                    .removeClass('d-none');
                $('#no-image').addClass('d-none');
            };
            reader.readAsDataURL(file);
        }
    });

    // Handle form submission
    $('#upload-form').submit(function(e) {
        e.preventDefault();
        
        // Show loading spinner
        $('#loading').removeClass('d-none');
        $('#detection-results, #classification-results, #recommendations').addClass('d-none');
        
        // Create form data
        const formData = new FormData();
        formData.append('image', $('#image-upload')[0].files[0]);
        
        // Send request to server
        $.ajax({
            url: '/analyze',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                // Hide loading spinner
                $('#loading').addClass('d-none');
                
                // Show results
                displayResults(response);
            },
            error: function(xhr, status, error) {
                // Hide loading spinner
                $('#loading').addClass('d-none');
                
                // Show error message
                alert('Error analyzing image: ' + error);
            }
        });
    });
});

function displayResults(results) {
    // Display detection results
    if (results.detections) {
        $('#detection-results').removeClass('d-none');
        
        // Show detection image
        const detectionImage = new Image();
        detectionImage.src = 'data:image/png;base64,' + results.detections.image;
        $('#detection-image').html(detectionImage);
        
        // Show detection stats
        const stats = results.detections.stats;
        let statsHtml = '<ul class="list-group">';
        statsHtml += `<li class="list-group-item">Total cavities detected: ${stats.total}</li>`;
        statsHtml += `<li class="list-group-item">Average confidence: ${(stats.avg_confidence * 100).toFixed(1)}%</li>`;
        statsHtml += '</ul>';
        $('#detection-stats').html(statsHtml);
    }
    
    // Display classification results
    if (results.severity) {
        $('#classification-results').removeClass('d-none');
        
        // Update confidence bar
        const confidence = results.confidence * 100;
        $('#confidence-bar')
            .css('width', confidence + '%')
            .attr('aria-valuenow', confidence);
        
        // Show severity text
        const severityClass = getSeverityClass(results.severity);
        const severityHtml = `
            <span class="severity-indicator ${severityClass}">
                ${results.severity.toUpperCase()}
            </span>
            <span>Confidence: ${confidence.toFixed(1)}%</span>
        `;
        $('#severity-text').html(severityHtml);
    }
    
    // Display recommendations
    if (results.recommendations) {
        $('#recommendations').removeClass('d-none');
        
        // Show recommendation list
        let recsHtml = '';
        results.recommendations.forEach(function(rec) {
            const urgencyClass = getUrgencyClass(rec.urgency);
            recsHtml += `
                <div class="recommendation-item ${urgencyClass} fade-in">
                    ${rec.text}
                </div>
            `;
        });
        $('#recommendation-list').html(recsHtml);
    }
}

function getSeverityClass(severity) {
    switch (severity.toLowerCase()) {
        case 'superficial':
            return 'severity-low';
        case 'medium':
            return 'severity-medium';
        case 'deep':
            return 'severity-high';
        default:
            return '';
    }
}

function getUrgencyClass(urgency) {
    switch (urgency.toLowerCase()) {
        case 'immediate':
            return 'urgent';
        case 'important':
            return 'warning';
        default:
            return '';
    }
}

// Helper function to format numbers
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

// Helper function to create charts (if needed)
function createChart(elementId, data) {
    // Implementation for charts if needed
    // This could use Chart.js or other charting libraries
} 