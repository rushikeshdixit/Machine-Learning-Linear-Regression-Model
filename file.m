d = importdata('iris.txt');
op = cellfun(@(x) split(x, ','), d, 'UniformOutput', false);
op1 = {};
for i = op'
    op1 = {op1{:} , i{1}{5}};
end
class_name = unique(op1);
op2 = [];
for index = 1:length(op')
    temp = string(find(cellfun(@(x) strcmp(x, op{index}{5}), class_name)));
    op{index} = str2double(op{index});
    op{index}(5) = temp;
    op2 = [op2; op{index}'];
end
% Part A - Linear regresion
Y = op2(:, end);
A = op2(:, 1:end-1);
beta = linear_regression(A, Y);
disp("Beta : " + join(string(beta'), ' '));
disp("Average Kfold Error : " + string(kfold_cross_valid(A, Y, 10)));
error = 0;
N = 1000;
k = 30;
for i = 1:N
    error = error + kfold_cross_valid(A, Y, k);
end
avg = error / N;
disp(avg);

% Part B - Classification
function [error] = kfold_cross_valid(A, Y, k)
    rperm = randperm(length(Y));
    size = length(Y)/k;
    error = 0;
    class_name = unique(Y);
    for i = 1:k
        test = rperm( size*(i-1)+1: size*i );
        train = setdiff( 1:length(Y),  test);
        beta = linear_regression(A(train, :), Y(train));
        Ycalc = classify(A(test,:), beta, length(class_name));
        error = error + sum_of_squared_error(Ycalc, Y(test));
    end
    error = error/k;
end

function [error] = sum_of_squared_error(Ycalc, Yact)
    error = Ycalc - Yact;
    error = sum(error .* error)/length(Ycalc);
end


function [Yop] = classify(test, beta, length_class)
   % No round -
   %Yop = test*beta;
   Yop = round(test*beta);
   Yop( Yop > length_class ) = length_class;
   Yop(Yop < 1) = 1;
end

%Part C - K-fold crossvalidation
function [beta] = linear_regression(A, Y)
    beta = inv(A'*A)*A'*Y;
end
