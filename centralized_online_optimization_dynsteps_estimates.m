% Compute step size is typically compute_step_size = @(t,D,G) D/(G*sqrt(t))

function [eval_X, eval_Y, eval_FX, eval_GX,  eval_inner_product_G_and_X , eval_prod_G, eval_FU, eval_GU, eval_dist_x_u, eval_dist_Xt_Xtplusone, eval_F_SAMPLING_POINTS, Xstore, wc]=...
                centralized_online_optimization_dynsteps_estimates(T,D,G,verbose,compute_step_size, NB_SAMPLES_LAMBDA)
    % Parameters
    % param.mu = 0.5;
    param_F.R = G; % /!\ param.R represents the radius over the norm of G and not the radius of the domain /!\
    function_class = 'ConvexBoundedGradient';
    % END - Parameters
    indices = cell(1,T+1);
    % PEP stuff
    P = pep();
    
    % Declaring multiple functions
    multiFunctions = @(x) P.DeclareFunction(function_class, param_F);
    F = foreach(multiFunctions,indices);
    
    % Declaring an indicator function
    param_id.D = D;
    id = P.DeclareFunction('ConvexIndicator',param_id);

    % Computing the reference point----------------
    % Aggregating functions together
    F_sum = F{1}; % Initializing the sum
    for t = 2:T
        F_sum = F_sum + F{t};
    end
    F_sum = F_sum + id; % Adding the indicator function
    
    [xs,fs]=F_sum.OptimalPoint(); % Computing the optima
    P.AddConstraint(xs^2==0); % Constraining optimal point to be zero
    %----------------------------------------------------------------------
    
    % Declaring a starting point
    x1=P.StartingPoint();
    %P.InitialCondition((x1-xs)^2<=D^2); % Add an initial condition ||x0-xs||^2<= 1
    %P.InitialCondition(x1^2==0);
    % (3) Algorithm
    F_fixed = cell(T,1); % f_t(x_t)
    G_fixed = cell(T,1); % grad_f_t(x_t)
    x=cell(T+1,1);% we store the iterates in a cell for convenience
    y=cell(T,1);
    % ---------------------------------------------------------------------
    x{1}= x1;
    for t=1:T
        [g,f,~]=F{t}.oracle(x{t}); % g=grad F(x), f=F(x)
        F_fixed{t}=f;
        G_fixed{t}=g;
        y{t} = x{t} - compute_step_size(t,D,G)* g; % Propagating 
        x{t+1} = projection_step(y{t},id);
    end

    % We evaluate f_t in the optimal point
    FU_fixed = cell(T,1); % f_t(u)
    GU_fixed = cell(T,1); % g_t(u)
    for t=1:T
        [GU_fixed{t},FU_fixed{t},~] = F{t}.oracle(xs);
        %P.AddConstraint(FU_fixed{t}==0); % 
    end
    %P.AddConstraint(FU_fixed{2}==0);
    
    % (4) Set up the performance metric
    F_sum_fixed = 0;
    for t=1:T % the oracle is called on 'F_t(x_t)' for all t from 1 to T
        %BEFORE: [g,f]=F{t}.oracle(x{t+1}); % g=grad F(x), f=F(x)
        id_val = id.value(x{t});
        % To be used into the Performance Metric
        F_sum_fixed=F_sum_fixed + F_fixed{t} + id_val;        
        % - - - 
    end
    [G_fixed{T+1},F_fixed{T+1}, ~] = F{T+1}.oracle(x{T+1});
    
    P.PerformanceMetric(F_sum_fixed - fs); % Worst-case evaluated as F(x)-F(xs)
    
    % Evaluating functionals in more points -------------------------------
    if NB_SAMPLES_LAMBDA > 0
        SAMPLING_POINTS = cell(T,NB_SAMPLES_LAMBDA); % Convex combinations between x_t and u 
        F_SAMPLING_POINTS = cell(T,NB_SAMPLES_LAMBDA); % Each f_t is evaluated over the convex combinations
        G_SAMPLING_POINTS = cell(T,NB_SAMPLES_LAMBDA); % Each nabla f_t is evaluated over the convex combinations 
        for t=1:T
            counter_lambda = 0;
            for lambda=linspace(0,1,NB_SAMPLES_LAMBDA)
                counter_lambda = counter_lambda + 1;
                SAMPLING_POINTS{t,counter_lambda} = x{t} * (1-lambda) + lambda*xs;
                [G_SAMPLING_POINTS{t,counter_lambda}, F_SAMPLING_POINTS{t,counter_lambda},~] = F{t}.oracle(SAMPLING_POINTS{t,counter_lambda});
            end
        end
    end
    
    % SOLVING THE PEP
    options = sdpsettings('verbose',max(verbose-1,0));
    P.TraceHeuristic(1);
    out=P.solve(verbose, options);

    % Post-treatment  -----------------------------------------------------
    if contains(out.solverDetails.info,"Successfully solved")
        wc=out.WCperformance;
    else
        if contains(out.solverDetails.info,"Unbounded objective")
            wc=-1;
        else
            wc=-2; % Other types of errors
        end
    end
    Xstore = zeros(T+1,length(double(x{t})));
    if verbose
        fprintf("Optimal point is: ");
        display(transpose(double(xs)));
        fprintf("Optimal value is: ")
        display(double(fs));
        for t=1:T+1
            fprintf("x%d",t);
            display(transpose(double(x{t})));
            %fprintf("g%d",t)
            %display(transpose(double(G_fixed{t})));
            %fprintf("y%d",t)
        end
    end
    for t=1:T+1
        Xstore(t,:) = transpose(double(x{t}));
    end
    % Retrieving estimates
    % ||X||_2
    eval_X = cell2mat(foreach(@(x_arg) sqrt(double(x_arg^2)), x));
    % ||x_t - x_t+1||
    eval_dist_Xt_Xtplusone = zeros(T,1);
    % ||x-u||_2
    eval_dist_x_u = cell2mat(foreach(@(x_arg) sqrt(double((x_arg - xs)^2)), x));
    % ||Y||_2
    eval_Y = cell2mat(foreach(@(y_arg) sqrt(double(y_arg^2)), y));
    % F_t(X_t)
    eval_FX = zeros(T,1);
    % ||gradient_F(X)||_2
    eval_GX = zeros(T,1);
    % f_t(x_t) - f_t(u)
    eval_FU = zeros(T,1); 
    % gradient_f_t(u) 
    eval_GU = zeros(T,1); % array of vectors
    for t=1:T
        eval_dist_Xt_Xtplusone(t) = sqrt(double((x{t}*x{t+1})));
        eval_FX(t) = double(F_fixed{t});
        eval_FU(t) = double(FU_fixed{t});
        eval_GX(t) = sqrt(double(G_fixed{t}^2));
        eval_GU(t) = sqrt(double(GU_fixed{t}^2));
    end
    % <g_t;x_t>
    eval_inner_product_G_and_X = zeros(T,1);
    for t=1:T+1
        eval_inner_product_G_and_X(t) = double(G_fixed{t}*x{t});
    end

    eval_prod_G = zeros(T);
    for t=1:T+1
        for k=1:T+1
            eval_prod_G(t,k) = double(G_fixed{t} * G_fixed{k});
        end
    end
    
    if NB_SAMPLES_LAMBDA>0
        eval_F_SAMPLING_POINTS = cell2mat(foreach(@(f_t) double(f_t),F_SAMPLING_POINTS)); % Each f_t is evaluated over the convex combinations
    else
        eval_F_SAMPLING_POINTS = eval_FX;
    end
    
    fprintf('PESTO bound: %d \n',wc);
end

