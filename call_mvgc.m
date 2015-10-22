load('xthetas.mat');

num_trials = 1000;
num_trials_at_a_time = 10;
num_obs = 1000;

mvgc_func(X(:, :, i), 1, num_obs);
gc12 = zeros(num_trials/num_trials_at_a_time, 1);
gc21 = zeros(num_trials/num_trials_at_a_time, 1);
j = 1;
%F = zeros(2, 2);
for i = 1:(num_trials/num_trials_at_a_time)
	[F, success] = mvgc_func(X(:, :, i:i+num_trials_at_a_time-1), num_trials_at_a_time, num_obs);
	if success
		gc12(j) = F(2, 1);
		gc21(j) = F(1, 2);
		j = j + 1;
	end
end

gc12 = gc12(1:j-1);
gc21 = gc21(1:j-1);

figure;
histogram(gc12);
hold on;
histogram(gc21);

figure;
plot(gc12, ones(size(gc12)), 'bx');
hold on;
plot(gc21, ones(size(gc12)), 'rx');
