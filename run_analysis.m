% Script to run the medical image disease detection system
% Make sure detectAndClassifyDisease.m is in the same directory as this script
% or in your MATLAB path

% Clear workspace and command window for a clean start
clear;
clc;

% Prompt user to select the medical image file
[filename, filepath] = uigetfile({'.jpg;.jpeg;.png;.tif;.tiff', 'Image Files (.jpg, *.jpeg, *.png, *.tif, *.tiff)'}, 'Select a medical image file');

% Check if user canceled the operation
if isequal(filename, 0) || isequal(filepath, 0)
    error('Image selection was canceled. Please run the script again and select an image.');
end

% Construct the full path to the image
imagePath = fullfile(filepath, filename);

% Display information
fprintf('Running medical image analysis on: %s\n', imagePath);
fprintf('Please wait, this may take a moment...\n\n');

% Call the disease detection function
try
    [diseaseName, affectedPercentage, diseaseComplexity, daysToCure] = detectAndClassifyDisease(imagePath);
    
    % Display a summary of results
    fprintf('\n===== SUMMARY OF RESULTS =====\n');
    fprintf('Disease detected: %s\n', diseaseName);
    fprintf('Affected area: %.2f%%\n', affectedPercentage);
    fprintf('Disease complexity: %s\n', diseaseComplexity);
    
    if daysToCure == -1
        fprintf('Estimated recovery time: Indeterminate, requires further investigation\n');
    else
        fprintf('Estimated days to cure: %d\n', daysToCure);
    end
    
    % Save results to a text file
    resultFile = [imagePath(1:end-4) '_results.txt'];
    fid = fopen(resultFile, 'w');
    fprintf(fid, 'Analysis Results for: %s\n', imagePath);
    fprintf(fid, '----------------------------\n');
    fprintf(fid, 'Disease: %s\n', diseaseName);
    fprintf(fid, 'Affected Area: %.2f%%\n', affectedPercentage);
    fprintf(fid, 'Complexity: %s\n', diseaseComplexity);
    if daysToCure == -1
        fprintf(fid, 'Estimated Days to Cure: Indeterminate\n');
    else
        fprintf(fid, 'Estimated Days to Cure: %d\n', daysToCure);
    end
    fclose(fid);
    
    fprintf('\nResults have been saved to: %s\n', resultFile);
    
catch ME
    % Handle any errors that occur during processing
    fprintf('\nError analyzing the image:\n%s\n', ME.message);
    fprintf('Please make sure all required MATLAB toolboxes are installed.\n');
end

% Optional: Create a simple batch processing function
function batchProcessImages()
    % Get folder containing medical images
    imageFolder = uigetdir('.', 'Select folder containing medical images');
    
    if imageFolder == 0
        fprintf('Batch processing canceled.\n');
        return;
    end
    
    % Get all image files
    imageFiles = dir(fullfile(imageFolder, '*.jpg'));
    imageFiles = [imageFiles; dir(fullfile(imageFolder, '*.jpeg'))];
    imageFiles = [imageFiles; dir(fullfile(imageFolder, '*.png'))];
    imageFiles = [imageFiles; dir(fullfile(imageFolder, '*.tif'))];
    imageFiles = [imageFiles; dir(fullfile(imageFolder, '*.tiff'))];
    
    if isempty(imageFiles)
        fprintf('No image files found in the selected folder.\n');
        return;
    end
    
    % Process each image
    fprintf('Found %d images to process.\n', length(imageFiles));
    
    % Create results table
    resultsTable = table('Size', [length(imageFiles), 5], ...
        'VariableTypes', {'string', 'string', 'double', 'string', 'double'}, ...
        'VariableNames', {'Filename', 'Disease', 'AffectedArea', 'Complexity', 'DaysToCure'});
    
    for i = 1:length(imageFiles)
        imagePath = fullfile(imageFolder, imageFiles(i).name);
        fprintf('Processing image %d of %d: %s\n', i, length(imageFiles), imageFiles(i).name);
        
        try
            [diseaseName, affectedPercentage, diseaseComplexity, daysToCure] = detectAndClassifyDisease(imagePath);
            
            % Store results in table
            resultsTable.Filename(i) = string(imageFiles(i).name);
            resultsTable.Disease(i) = string(diseaseName);
            resultsTable.AffectedArea(i) = affectedPercentage;
            resultsTable.Complexity(i) = string(diseaseComplexity);
            resultsTable.DaysToCure(i) = daysToCure;
            
            fprintf('  - Found: %s (%.2f%% affected)\n', diseaseName, affectedPercentage);
        catch ME
            fprintf('  - Error processing this image: %s\n', ME.message);
            resultsTable.Filename(i) = string(imageFiles(i).name);
            resultsTable.Disease(i) = "ERROR";
        end
    end
    
    % Save results to CSV
    resultsPath = fullfile(imageFolder, 'batch_results.csv');
    writetable(resultsTable, resultsPath);
    fprintf('\nBatch processing complete. Results saved to:\n%s\n', resultsPath);
    
    % Display summary
    fprintf('\nSummary of findings:\n');
    diseaseCount = groupcounts(resultsTable, 'Disease');
    disp(diseaseCount);
end