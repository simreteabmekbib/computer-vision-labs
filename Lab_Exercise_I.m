% Lab_Exercise_I.m
A = [1 2; 3 4];
B = [5 6; 7 8];
[m, n] = size(A);
[n_check, p] = size(B);
if n ~= n_check
    error('Matrix dimensions incompatible for multiplication');
end
C = zeros(m, p);
for i = 1:m
    for j = 1:p
        for k = 1:n
            C(i, j) = C(i, j) + A(i, k) * B(k, j);
        end
    end
end
disp(C);

function root = bisection_root(f, a, b, tol)
    if f(a) * f(b) > 0
        error('Function values at a and b must have opposite signs');
    end
    while (b - a) > tol
        c = (a + b) / 2;
        if f(a) * f(c) < 0
            b = c;
        else
            a = c;
        end
    end
    root = (a + b) / 2;
end

f = @(x) x^2 - 2;  % Root of sqrt(2)
root = bisection_root(f, 1, 2, 1e-6);
disp(root);

%Bubble Sort
function sorted = bubble_sort(arr)
    n = length(arr);
    for i = 1:n-1
        for j = 1:n-i
            if arr(j) > arr(j+1)
                temp = arr(j);
                arr(j) = arr(j+1);
                arr(j+1) = temp;
            end
        end
    end
    sorted = arr;
end

%Merge Sort
function sorted = merge_sort(arr)
    if length(arr) <= 1
        sorted = arr;
        return;
    end
    mid = floor(length(arr)/2);
    left = merge_sort(arr(1:mid));
    right = merge_sort(arr(mid+1:end));
    sorted = merge(left, right);
end

%Merge Sort
function merged = merge(left, right)
    i = 1; j = 1; merged = [];
    while i <= length(left) && j <= length(right)
        if left(i) <= right(j)
            merged = [merged, left(i)];
            i = i + 1;
        else
            merged = [merged, right(j)];
            j = j + 1;
        end
    end
    merged = [merged, left(i:end), right(j:end)];
end

%Quick Sort
function sorted = quicksort(arr)
    if length(arr) <= 1
        sorted = arr;
        return;
    end
    pivot = arr(1);
    left = [];
    right = [];
    for i = 2:length(arr)
        if arr(i) < pivot
            left = [left, arr(i)];
        else
            right = [right, arr(i)];
        end
    end
    sorted = [quicksort(left), pivot, quicksort(right)];
end

arr = [5, 3, 8, 4, 2];
disp(bubble_sort(arr));
disp(merge_sort(arr));
disp(quicksort(arr));

function idx = binary_search(arr, target)
    low = 1;
    high = length(arr);
    idx = -1;
    while low <= high
        mid = floor((low + high) / 2);
        if arr(mid) == target
            idx = mid;
            return;
        elseif arr(mid) < target
            low = mid + 1;
        else
            high = mid - 1;
        end
    end
end

sorted_arr = [1, 3, 5, 7, 9];
idx = binary_search(sorted_arr, 5);
disp(idx);  % 3 (1-based)

function res = factorial(n)
    if n <= 1
        res = 1;
    else
        res = n * factorial(n - 1);
    end
end

disp(factorial(5));

function bool = is_palindrome(str)
    str = lower(regexprep(str, '[^a-zA-Z]', ''));  % Remove non-letters, lowercase
    n = length(str);
    bool = true;
    for i = 1:floor(n/2)
        if str(i) ~= str(n - i + 1)
            bool = false;
            return;
        end
    end
end
disp(is_palindrome("A man a plan a canal Panama"));  % true

function [mean_val, median_val, mode_val] = stats(data)
    mean_val = sum(data) / length(data);
    sorted_data = sort(data);
    n = length(data);
    if mod(n, 2) == 0
        median_val = (sorted_data(n/2) + sorted_data(n/2 + 1)) / 2;
    else
        median_val = sorted_data((n+1)/2);
    end
    % Mode: Simple frequency count
    unique_vals = unique(data);
    counts = histc(data, unique_vals);
    [~, max_idx] = max(counts);
    mode_val = unique_vals(max_idx);
end
data = [1, 2, 2, 3, 4];
[meanv, medianv, modev] = stats(data);
disp([meanv, medianv, modev]);
