% preprocess
function output = preprocess(input)
output = sign(input) .* abs(input) .^ 0.5;
output = yael_vecs_normalize(output,2,0);
end