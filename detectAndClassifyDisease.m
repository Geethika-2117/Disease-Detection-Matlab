function [diseaseName, affectedPercentage, diseaseComplexity, daysToCure] = detectAndClassifyDisease(inputImagePath)
    % DETECTANDCLASSIFYDISEASE Analyzes medical images to detect and classify diseases
    %
    % This function uses matrix decomposition techniques to detect abnormalities in 
    % medical images and classify the detected disease
    %
    % Inputs:
    %   inputImagePath - Path to the medical image file (string)
    %
    % Outputs:
    %   diseaseName - Name of the detected disease (string)
    %   affectedPercentage - Percentage of the affected region (double)
    %   diseaseComplexity - Complexity level of the disease (string: 'Low', 'Moderate', 'High', 'Critical')
    %   daysToCure - Estimated days required for treatment (integer)
    
    % Check if input path exists
    if ~exist(inputImagePath, 'file')
        error('Input image file does not exist. Please provide a valid path.');
    end
    
    % Read the input medical image
    try
        img = imread(inputImagePath);
        fprintf('Successfully loaded image: %s\n', inputImagePath);
    catch
        error('Error reading the image. Please ensure it is a valid image file.');
    end
    
    % Convert to grayscale if the image is RGB
    if size(img, 3) == 3
        img = rgb2gray(img);
        fprintf('Converted RGB image to grayscale for processing.\n');
    end
    
    % Convert to double for mathematical operations with range checking
    img = double(img);
    if max(img(:)) > 255
        img = img / max(img(:)) * 255;
    end
    
    % Display original image
    figure('Name', 'Medical Image Disease Detection and Classification');
    subplot(2, 4, 1);
    imshow(uint8(img));
    title('Original Image');
    
    % Apply image enhancement using CLAHE or standard histogram equalization
    try
        enhanced_img = adapthisteq(uint8(img));
    catch
        warning('CLAHE failed, using standard histogram equalization instead.');
        enhanced_img = histeq(uint8(img));
    end
    enhanced_img = double(enhanced_img);
    
    subplot(2, 4, 2);
    imshow(uint8(enhanced_img));
    title('Enhanced Image');
    
    % Apply edge-preserving noise reduction
    filtered_img = imgaussfilt(enhanced_img, 1.2);
    
    subplot(2, 4, 3);
    imshow(uint8(filtered_img));
    title('Filtered Image');
    
    % Handle large images before SVD to prevent memory issues
    [rows, cols] = size(filtered_img);
    max_dimension = 1000;
    
    % Perform SVD with downsampling if necessary
    if rows > max_dimension || cols > max_dimension
        scale_factor = max_dimension / max(rows, cols);
        filtered_img_small = imresize(filtered_img, scale_factor);
        fprintf('Image downsampled for SVD processing (%.1f%% of original size)\n', scale_factor * 100);
        [U, S, V] = svd(filtered_img_small);
    else
        [U, S, V] = svd(filtered_img);
    end
    
    % Analyze singular values
    singular_values = diag(S);
    
    % Plot singular values
    subplot(2, 4, 4);
    semilogy(singular_values, 'o-');
    title('Singular Values');
    xlabel('Index');
    ylabel('Value (log scale)');
    grid on;
    
    % Determine optimal rank for reconstruction (use energy preservation criterion)
    total_energy = sum(singular_values.^2);
    energy_ratio = cumsum(singular_values.^2) / total_energy;
    k = find(energy_ratio >= 0.9, 1); % Preserve 90% energy
    
    % Low-rank approximation (disease pattern isolation)
    S_reduced = zeros(size(S));
    S_reduced(1:k, 1:k) = S(1:k, 1:k);
    low_rank_img = U * S_reduced * V';
    
    % High-frequency components (anomalies)
    S_residual = zeros(size(S));
    S_residual(k+1:end, k+1:end) = S(k+1:end, k+1:end);
    residual_img = U * S_residual * V';
    
    % If we downsampled for SVD, we need to resize results back to original dimensions
    if rows > max_dimension || cols > max_dimension
        low_rank_img = imresize(low_rank_img, [rows, cols]);
        residual_img = imresize(residual_img, [rows, cols]);
    end
    
    % Enhance residual image to highlight anomalies
    residual_enhanced = residual_img ./ (max(residual_img(:)) + eps) * 255;
    
    subplot(2, 4, 5);
    imshow(uint8(low_rank_img));
    title('Normal Tissue Structure');
    
    subplot(2, 4, 6);
    imshow(uint8(residual_enhanced));
    title('Anomaly Map');
    
    % Apply thresholding to identify potential disease regions
    try
        threshold = calculateOtsuThreshold(residual_enhanced(:));
    catch
        % Fallback to simple thresholding if Otsu's method fails
        warning('Otsu thresholding failed, using simple thresholding instead.');
        threshold = mean(residual_enhanced(:)) + std(residual_enhanced(:));
    end
    binary_mask = residual_enhanced > threshold;
    
    % Apply morphological operations with error handling
    try
        binary_mask = imopen(binary_mask, strel('disk', 2));
        binary_mask = imclose(binary_mask, strel('disk', 4));
        binary_mask = imfill(binary_mask, 'holes');
    catch ME
        warning('Morphological operations failed: %s. Using unprocessed binary mask.', E.message);
    end
    
    % Label the connected components and measure properties
    [labeled_mask, num_regions] = bwlabel(binary_mask);
    region_props = regionprops(labeled_mask, residual_enhanced, ...
        'Area', 'Centroid', 'MajorAxisLength', 'MinorAxisLength', ...
        'Orientation', 'Eccentricity', 'MeanIntensity', 'MaxIntensity');
    
    % Filter out small regions (likely noise)
    min_area = 20;
    valid_regions = find([region_props.Area] > min_area);
    num_valid_regions = length(valid_regions);
    
    % Calculate total affected area
    total_pixels = numel(img);
    affected_pixels = sum(binary_mask(:));
    affectedPercentage = (affected_pixels / total_pixels) * 100;
    
    % Overlay detected regions on original image
    overlay_img = repmat(uint8(img), [1, 1, 3]);
    
    % Mark potential disease regions in red
    for i = 1:length(valid_regions)
        region_idx = valid_regions(i);
        region_mask = (labeled_mask == region_idx);
        [rows, cols] = find(region_mask);
        
        for j = 1:length(rows)
            overlay_img(rows(j), cols(j), 1) = 255;  % Red channel
            overlay_img(rows(j), cols(j), 2) = 0;    % Green channel
            overlay_img(rows(j), cols(j), 3) = 0;    % Blue channel
        end
        
        % Draw region number and centroid
        if ~isempty(rows) && ~isempty(cols)
            centroid = region_props(region_idx).Centroid;
            try
                overlay_img = insertText(overlay_img, centroid, num2str(i), ...
                    'BoxOpacity', 0, 'TextColor', 'yellow', 'FontSize', 12);
            catch
                % If insertText fails, skip labeling
                warning('Could not insert text label for region %d', i);
            end
        end
    end
    
    subplot(2, 4, 7);
    imshow(overlay_img);
    title(['Detected Regions: ', num2str(num_valid_regions)]);
    
    % Extract features for disease classification
    if num_valid_regions > 0
        % Feature extraction
        features = extractDiseaseFeatures(img, binary_mask, region_props, valid_regions, residual_enhanced);
        
        % Disease classification based on extracted features
        [diseaseName, confidence, diseaseComplexity, daysToCure] = classifyDisease(features);
        
        % Display classification results
        result_text = sprintf('%s (%.1f%% confidence)\nAffected area: %.2f%%\nComplexity: %s\nDays to cure: %d', ...
            diseaseName, confidence, affectedPercentage, diseaseComplexity, daysToCure);
        
        subplot(2, 4, 8);
        imshow(zeros(size(img))+240);
        text(size(img,2)/2, size(img,1)/2, result_text, ...
            'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
        title('Classification Result');
        
        % Print detailed analysis
        fprintf('\n===== DISEASE DETECTION RESULTS =====\n');
        fprintf('Detected disease: %s (%.1f%% confidence)\n', diseaseName, confidence);
        fprintf('Total affected area: %.2f%%\n', affectedPercentage);
        fprintf('Disease complexity: %s\n', diseaseComplexity);
        fprintf('Estimated days to cure: %d\n', daysToCure);
        fprintf('Number of affected regions: %d\n', num_valid_regions);
        
        % Report on each significant region
        fprintf('\nDetailed region analysis:\n');
        for i = 1:length(valid_regions)
            region_idx = valid_regions(i);
            region_area_percent = (region_props(region_idx).Area / total_pixels) * 100;
            fprintf('Region %d: %.2f%% of image area, Max intensity: %.1f\n', ...
                i, region_area_percent, region_props(region_idx).MaxIntensity);
        end
    else
        diseaseName = 'No disease detected';
        affectedPercentage = 0;
        diseaseComplexity = 'None';
        daysToCure = 0;
        
        subplot(2, 4, 8);
        imshow(zeros(size(img))+240);
        text(size(img,2)/2, size(img,1)/2, 'No disease detected', ...
            'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
        title('Classification Result');
        
        fprintf('\n===== DISEASE DETECTION RESULTS =====\n');
        fprintf('No significant disease markers detected\n');
    end
    
    % Save results
    [pathstr, name, ~] = fileparts(inputImagePath);
    result_path = fullfile(pathstr, [name '_analyzed.png']);
    saveas(gcf, result_path);
    fprintf('\nAnalysis results saved to: %s\n', result_path);
end

function features = extractDiseaseFeatures(original_img, binary_mask, region_props, valid_regions, residual_img)
    % Extract features for disease classification
    
    % Image-level features
    img_size = numel(original_img);
    affected_ratio = sum(binary_mask(:)) / img_size;
    
    % Calculate texture features using GLCM with error handling
    try
        % First ensure image is in proper range for graycomatrix
        img_for_glcm = uint8(original_img);
        
        % Use smaller offsets to handle potential memory issues
        glcm = graycomatrix(img_for_glcm, 'Offset', [0 1; -1 0]);
        glcm_stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
    catch ME
        warning('GLCM calculation failed: %s. Using default texture values.', E.message);
        % Create a struct with default values
        glcm_stats = struct('Contrast', 0, 'Correlation', 0, 'Energy', 1, 'Homogeneity', 1);
    end
    
    % Region features
    if ~isempty(valid_regions)
        % Area statistics
        areas = [region_props(valid_regions).Area];
        mean_area = mean(areas);
        std_area = std(areas);
        max_area = max(areas);
        
        % Shape features
        eccentricities = [region_props(valid_regions).Eccentricity];
        mean_eccentricity = mean(eccentricities);
        
        axes_ratio = [];
        for i = 1:length(valid_regions)
            region_idx = valid_regions(i);
            if region_props(region_idx).MajorAxisLength > 0
                axes_ratio(end+1) = region_props(region_idx).MinorAxisLength / ...
                    region_props(region_idx).MajorAxisLength;
            else
                axes_ratio(end+1) = 1;
            end
        end
        mean_axes_ratio = mean(axes_ratio);
        
        % Intensity features
        intensities = [region_props(valid_regions).MeanIntensity];
        mean_intensity = mean(intensities);
        max_intensity = max([region_props(valid_regions).MaxIntensity]);
        
        % Spatial distribution
        centroids = cat(1, region_props(valid_regions).Centroid);
        if size(centroids, 1) > 1
            try
                centroid_distances = pdist(centroids);
                avg_distance = mean(centroid_distances);
                max_distance = max(centroid_distances);
            catch
                avg_distance = 0;
                max_distance = 0;
            end
        else
            avg_distance = 0;
            max_distance = 0;
        end
    else
        % Default values if no valid regions
        mean_area = 0; std_area = 0; max_area = 0;
        mean_eccentricity = 0; mean_axes_ratio = 0;
        mean_intensity = 0; max_intensity = 0;
        avg_distance = 0; max_distance = 0;
    end
    
    % Pack all features into a structure
    features = struct();
    features.affected_ratio = affected_ratio;
    features.contrast = mean(glcm_stats.Contrast);
    features.correlation = mean(glcm_stats.Correlation);
    features.energy = mean(glcm_stats.Energy);
    features.homogeneity = mean(glcm_stats.Homogeneity);
    features.mean_area = mean_area;
    features.std_area = std_area;
    features.max_area = max_area;
    features.mean_eccentricity = mean_eccentricity;
    features.mean_axes_ratio = mean_axes_ratio;
    features.mean_intensity = mean_intensity;
    features.max_intensity = max_intensity;
    features.avg_distance = avg_distance;
    features.max_distance = max_distance;
    features.num_regions = length(valid_regions);
    
    % SVD-based features
    try
        [~, S, ~] = svd(double(binary_mask));
        singular_values = diag(S);
        if length(singular_values) >= 3
            features.sv_ratio1 = singular_values(1) / sum(singular_values);
            features.sv_ratio2 = singular_values(2) / singular_values(1);
            features.sv_ratio3 = singular_values(3) / singular_values(2);
        else
            features.sv_ratio1 = 1;
            features.sv_ratio2 = 0;
            features.sv_ratio3 = 0;
        end
    catch
        % Default SVD values if calculation fails
        features.sv_ratio1 = 1;
        features.sv_ratio2 = 0;
        features.sv_ratio3 = 0;
    end
    
    % Calculate radial distribution of anomalies
    [rows, cols] = find(binary_mask);
    if ~isempty(rows) && ~isempty(cols)
        center_y = size(binary_mask, 1) / 2;
        center_x = size(binary_mask, 2) / 2;
        distances = sqrt((rows - center_y).^2 + (cols - center_x).^2);
        features.mean_distance_from_center = mean(distances);
        features.std_distance_from_center = std(distances);
    else
        features.mean_distance_from_center = 0;
        features.std_distance_from_center = 0;
    end
end

function [diseaseName, confidence, diseaseComplexity, daysToCure] = classifyDisease(features)
    % Classify disease based on extracted features
    % This is a rule-based classifier that now includes complexity and days to cure
    
    % Extract key features for classification
    affected_ratio = features.affected_ratio;
    num_regions = features.num_regions;
    eccentricity = features.mean_eccentricity;
    axes_ratio = features.mean_axes_ratio;
    contrast = features.contrast;
    energy = features.energy;
    sv_ratio1 = features.sv_ratio1;
    mean_distance_from_center = features.mean_distance_from_center;
    
    % Rule-based classification with added complexity and days to cure
    if affected_ratio < 0.005
        % Very small affected area
        diseaseName = 'No significant pathology';
        confidence = 90;
        diseaseComplexity = 'None';
        daysToCure = 0;
    elseif (affected_ratio > 0.2) && (num_regions <= 3) && (contrast > 20)
        % Large single or few regions with high contrast
        diseaseName = 'Pneumonia';
        confidence = 85 + min(10, 50 * affected_ratio);
        
        % Complexity and days to cure based on affected ratio and contrast
        if affected_ratio > 0.4 && contrast > 30
            diseaseComplexity = 'High';
            daysToCure = 21 + round(10 * affected_ratio);
        elseif affected_ratio > 0.3
            diseaseComplexity = 'Moderate';
            daysToCure = 14 + round(7 * affected_ratio);
        else
            diseaseComplexity = 'Moderate';
            daysToCure = 10 + round(5 * affected_ratio);
        end
    elseif (affected_ratio > 0.03) && (affected_ratio < 0.15) && (num_regions >= 3) && (sv_ratio1 > 0.7)
        % Multiple distinct regions with high primary singular value
        diseaseName = 'Tuberculosis';
        confidence = 75 + 10 * sv_ratio1;
        
        % TB is typically complex and requires long treatment
        diseaseComplexity = 'High';
        daysToCure = 180 + round(20 * num_regions);
    elseif (num_regions > 5) && (affected_ratio < 0.1) && (eccentricity < 0.7)
        % Multiple small, round regions
        diseaseName = 'Lung Nodules';
        confidence = 80 - 100 * abs(0.5 - axes_ratio);
        
        if num_regions > 10
            diseaseComplexity = 'Critical';
            daysToCure = 90 + round(5 * num_regions); % Often requires surgery or long-term treatment
        else
            diseaseComplexity = 'Moderate';
            daysToCure = 60 + round(3 * num_regions);
        end
    elseif (affected_ratio > 0.1) && (energy < 0.2) && (mean_distance_from_center > 50)
        % Diffuse changes in periphery with low energy
        diseaseName = 'Pulmonary Fibrosis';
        confidence = 70 + 20 * (1 - energy);
        
        %Fibrosis is often chronic and difficult to treat
        diseaseComplexity = 'High';
        daysToCure = 120 + round(60 * affected_ratio);
    elseif (affected_ratio > 0.15) && (num_regions <= 2) && (eccentricity > 0.8)
        % Large elongated region
        diseaseName = 'Atelectasis';
        confidence = 75 + 15 * eccentricity;
        
        diseaseComplexity = 'Moderate';
        daysToCure = 7 + round(7 * affected_ratio);
    elseif (affected_ratio > 0.25) && (energy < 0.3) && (contrast > 15)
        % Very large affected area with decent contrast
        diseaseName = 'Pleural Effusion';
        confidence = 80 + 15 * affected_ratio;
        
        if affected_ratio > 0.4
            diseaseComplexity = 'High';
            daysToCure = 14 + round(10 * affected_ratio);
        else
            diseaseComplexity = 'Moderate';
            daysToCure = 10 + round(5 * affected_ratio);
        end
    elseif (affected_ratio > 0.08) && (num_regions >= 4) && (sv_ratio1 < 0.6)
        % Multiple regions with distributed singular values
        diseaseName = 'Metastatic Disease';
        confidence = 65 + 20 * (1 - sv_ratio1);
        
        % Metastatic disease is serious and requires complex treatment
        diseaseComplexity = 'Critical';
        daysToCure = 180 + round(30 * num_regions); % Often requires long-term treatment
    elseif (affected_ratio > 0.02) && (affected_ratio < 0.06) && (eccentricity < 0.5)
        % Small circular region
        diseaseName = 'Solitary Pulmonary Nodule';
        confidence = 75 + 40 * (1 - eccentricity);
        
        if affected_ratio > 0.04
            diseaseComplexity = 'Moderate';
            daysToCure = 30 + round(200 * affected_ratio);
        else
            diseaseComplexity = 'Low';
            daysToCure = 14 + round(100 * affected_ratio);
        end
    else
        % Default case for unknown patterns
        diseaseName = 'Abnormal finding - further investigation needed';
        confidence = 60;
        diseaseComplexity = 'Unknown';
        daysToCure = -1; %-1 indicates indeterminate
    end
end

function threshold = calculateOtsuThreshold(grayLevels)
    % Otsu's method for automatic thresholding
    % Input: grayLevels - vector of gray level intensities
    
    % Calculate normalized histogram
    [counts, ~] = histcounts(grayLevels, 256);
    p = counts / sum(counts);
    
    % Initialize variables
    maxVariance = 0;
    threshold = 0;
    
    % Iterate through all possible thresholds
    for t = 1:255
        % Probability of background class
        w0 = sum(p(1:t));
        if w0 == 0
            continue;
        end
        
        % Probability of foreground class
        w1 = 1 - w0;
        if w1 == 0
            continue;
        end
        
        % Calculate class means
        mu0 = sum((0:t-1)' .* p(1:t)) / w0;
        mu1 = sum((t:255)' .* p(t+1:256)) / w1;
        
        % Calculate between-class variance
        variance = w0 * w1 * (mu0 - mu1)^2;
        
        % Update threshold if a higher variance is found
        if variance > maxVariance
            maxVariance = variance;
            threshold = t;
        end
    end
end