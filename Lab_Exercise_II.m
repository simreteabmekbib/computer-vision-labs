% Lab_Exercise_II.m
dir_path = './faces/'; % Directory containing images
max_display_size = 500; % Default maximum display size

% Supported image extensions
extensions = {'*.png', '*.jpg', '*.jpeg', '*.bmp'};

% Collect all image files
files = [];
for ext = extensions
    files = [files; dir(fullfile(dir_path, ext{:}))]; %#ok<AGROW>
end

if isempty(files)
    error('No supported images (PNG, JPG, JPEG, BMP) found in %s', dir_path);
end

% Initialize figure
num_files = length(files);
figure('Name', 'Image Display with Colormaps', 'NumberTitle', 'off');

% Process each image
for i = 1:num_files
    try
        % Read image
        img_path = fullfile(dir_path, files(i).name);
        img = imread(img_path);

        % Convert to grayscale if needed
        if size(img, 3) == 3
            img_gray = rgb2gray(img);
        else
            img_gray = double(img);
        end

        % Resize image for display if too large
        [h, w] = size(img_gray);
        scale = min(1, max_display_size / max(h, w));
        if scale < 1
            img_gray = imresize(img_gray, scale);
        end

        % Display with different colormaps
        subplot(num_files, 3, (i-1)*3 + 1); imshow(img_gray, []); colormap(gray); title([files(i).name ' - Gray']);
        subplot(num_files, 3, (i-1)*3 + 2); imshow(img_gray, []); colormap(hot); title([files(i).name ' - Hot']);
        subplot(num_files, 3, (i-1)*3 + 3); imshow(img_gray, []); colormap(jet); title([files(i).name ' - Jet']);

    catch err
        fprintf('Error processing %s: %s\n', files(i).name, err.message);
    end
end

% Adjust figure layout
set(gcf, 'Position', [100, 100, 1200, min(800, num_files * 200)]);

% Lab_Exercise_III.m
input_path = 'man.png'; % Path to the input image
target_width = 256; % Desired width
target_height = 256; % Desired height

% Read the image
img = imread(input_path);

% Check if image is RGB or grayscale
if size(img, 3) == 3
    is_rgb = true;
else
    is_rgb = false;
end

% Get original dimensions
[orig_h, orig_w, ~] = size(img);
aspect_ratio = orig_w / orig_h;
target_aspect = target_width / target_height;

% Calculate new dimensions while maintaining aspect ratio
if aspect_ratio > target_aspect
    new_h = target_height;
    new_w = round(target_height * aspect_ratio);
else
    new_w = target_width;
    new_h = round(target_width / aspect_ratio);
end

% Resize the image
resized = imresize(img, [new_h, new_w]);

% For PSNR, resize the resized image back to original dimensions for comparison
resized_for_psnr = imresize(resized, [orig_h, orig_w]);

% Display original and resized images
figure;
subplot(1, 2, 1); imshow(img); title(sprintf('Original: %dx%d', orig_w, orig_h));
subplot(1, 2, 2); imshow(resized); title(sprintf('Resized: %dx%d', new_w, new_h));

% Calculate PSNR (Peak Signal-to-Noise Ratio)
mse = mean((double(img(:)) - double(resized_for_psnr(:))).^2);
if mse == 0
    psnr_val = Inf; % Perfect match
else
    psnr_val = 10 * log10((255^2) / mse);
end
disp(['PSNR: ' num2str(psnr_val) ' dB']);

% Lab_Exercise_IV.m
img_path = 'peppers.tiff'; % Path to the input color image

% Read the image with error handling
try
    img = imread(img_path);
catch err
    error('Failed to read %s: %s', img_path, err.message);
end

% Check if image is color (RGB)
if size(img, 3) == 3
    R = double(img(:,:,1));
    G = double(img(:,:,2));
    B = double(img(:,:,3));

    % Average method
    gray_avg = uint8((R + G + B) / 3);

    % Luminosity method (ITU-R BT.601 weights)
    gray_lum = uint8(0.21*R + 0.72*G + 0.07*B);

    % Desaturation method
    gray_desat = uint8((max(max(R,G),B) + min(min(R,G),B)) / 2);

    % Display images
    figure('Name', 'Grayscale Conversions', 'NumberTitle', 'off');
    subplot(2,2,1); imshow(img); title('Original Color');
    subplot(2,2,2); imshow(gray_avg); title('Average');
    subplot(2,2,3); imshow(gray_lum); title('Luminosity');
    subplot(2,2,4); imshow(gray_desat); title('Desaturation');

    % Compare: Variance (higher = more detail preservation)
    var_avg = var(double(gray_avg(:)));
    var_lum = var(double(gray_lum(:)));
    var_desat = var(double(gray_desat(:)));
    disp(['Variances - Avg: ' num2str(var_avg) ', Lum: ' num2str(var_lum) ', Desat: ' num2str(var_desat)]);
else
    error('Input must be a color (RGB) image');
end

function cropped = crop_image(img_path, x1, y1, x2, y2)
    % Crop image to region (x1,y1) to (x2,y2)
    img = imread(img_path);
    if x1 < 0 || y1 < 0 || x2 > size(img, 2) || y2 > size(img, 1) || x1 >= x2 || y1 >= y2
        error('Invalid coordinates');
    end
    cropped = img(y1:y2, x1:x2, :);
    figure; imshow(cropped); title('Cropped Image');
end
crop_image('man.png', 100, 100, 300, 300);

function resize_with_interp(img_path, width, height)
    % Compare nearest, bilinear, bicubic interpolation
    img = imread(img_path);
    nearest = imresize(img, [height width], 'nearest');
    bilinear = imresize(img, [height width], 'bilinear');
    bicubic = imresize(img, [height width], 'bicubic');

    figure;
    subplot(1,3,1); imshow(nearest); title('Nearest');
    subplot(1,3,2); imshow(bilinear); title('Bilinear');
    subplot(1,3,3); imshow(bicubic); title('Bicubic');

    % Efficiency: Rough timing
    tic; imresize(img, [height width], 'nearest'); t1 = toc;
    tic; imresize(img, [height width], 'bilinear'); t2 = toc;
    tic; imresize(img, [height width], 'bicubic'); t3 = toc;
    disp(['Times (s): Nearest ' num2str(t1) ', Bilinear ' num2str(t2) ', Bicubic ' num2str(t3)]);
end
resize_with_interp('man.png', 256, 256);

function rotated = rotate_image(img_path, angle)
    % ROTATE_IMAGE - Rotate an image by a specified angle
    % Inputs:
    %   img_path: Path to the input image (e.g., 'man.png')
    %   angle: Rotation angle in degrees (positive = counterclockwise)
    % Outputs:
    %   rotated: The rotated image
    % Displays original and rotated images

    % Read the image with error handling
    try
        img = imread(img_path);
    catch err
        error('Failed to read %s: %s', img_path, err.message);
    end

    % Get original dimensions
    [h, w, ~] = size(img);

    % Rotate the image with 'bilinear' interpolation and 'crop' to keep original size
    rotated = imrotate(img, angle, 'bilinear', 'crop');

    % Display original and rotated images
    figure('Name', 'Image Rotation', 'NumberTitle', 'off');
    subplot(1, 2, 1); imshow(img); title('Original');
    subplot(1, 2, 2); imshow(rotated); title(['Rotated ' num2str(angle) '°']);
end
rotate_image('man.png', 45);

function download_images(url_list, save_dir)
    % Download images from URLs, handle formats
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end
    for i = 1:length(url_list)
        try
            [~, name, ext] = fileparts(url_list{i});
            urlwrite(url_list{i}, fullfile(save_dir, [name ext]));
            img = imread(fullfile(save_dir, [name ext]));
            figure; imshow(img); title([name ext]);
        catch err
            disp(['Error downloading ' url_list{i} ': ' err.message]);
        end
    end
end
urls = {'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRInKFKLo7HSnAQ5DGzi7s0OwOSoQ7qxFHyrA&s', 'https://cdn.britannica.com/16/234216-050-C66F8665/beagle-hound-dog.jpg'};
download_images(urls, './downloaded/');

% Lab_Exercise_VI.m
brightness = 50; % Brightness adjustment (0-100)
contrast = 30; % Contrast adjustment (0-100)

% Validate inputs
if ~isnumeric(brightness) || brightness < 0 || brightness > 100
    error('Brightness must be a number between 0 and 100');
end
if ~isnumeric(contrast) || contrast < 0 || contrast > 100
    error('Contrast must be a number between 0 and 100');
end

% Initialize camera
try
    cam = videoinput('winvideo', 1); % Adjust adapter and device ID as needed
catch err
    error('Failed to initialize camera: %s', err.message);
end

% Set camera properties
try
    set(cam, 'Brightness', brightness, 'Contrast', contrast);
catch err
    delete(cam);
    clear cam;
    error('Failed to set camera properties: %s', err.message);
end

% Capture image
try
    img = getsnapshot(cam);
catch err
    delete(cam);
    clear cam;
    error('Failed to capture image: %s', err.message);
end

% Clean up camera
delete(cam);
clear cam;

% Display image
figure('Name', 'Captured Image', 'NumberTitle', 'off');
imshow(img); title(sprintf('Captured Image (Brightness: %d, Contrast: %d)', brightness, contrast));

function display_color_spaces(img_path)
    % Display RGB and CMYK color spaces
    img_rgb = imread(img_path);
    img_cmyk = rgb2cmyk(img_rgb);  % Requires image package extension
    figure;
    subplot(1,2,1); imshow(img_rgb); title('RGB');
    subplot(1,2,2); imshow(img_cmyk); title('CMYK');
end
display_color_spaces('man.png');

function extract_metadata(img_path)
    % Extract creation date, resolution, color space
    info = imfinfo(img_path);
    disp(['Resolution: ' num2str(info.Width) 'x' num2str(info.Height)]);
    disp(['Date: ' info.DateTime]);
    disp(['Color Type: ' info.ColorType]);
end
extract_metadata('man.png');

function adjusted = adjust_pixels(img_path, scale, offset)
    % Adjust pixel values with scale and offset
    img = double(imread(img_path));
    adjusted = (img * scale) + offset;
    adjusted = max(0, min(255, adjusted));  % Clamp to [0,255]
    figure; imshow(uint8(adjusted)); title('Adjusted Image');
end
adjust_pixels('man.png', 1.2, 20);
