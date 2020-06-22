function pc = beta_dist(n_correct, im_col)

a = n_correct + 1;
b = (length(im_col)-n_correct) + 1;

pc = betarnd(a, b);

end