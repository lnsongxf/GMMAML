#using LinearAlgebra
using Econometrics

# dgp: CLRM with collinearity and irrelevant regressors
function dgp(n::Int64)
    # create covariances of regressors
    P = 2*rand(10,10) .- 1
    for i in 1:10 P[i,i] = 1.0 end
    P = UpperTriangular(P)
    # generate data and parameters
    x = [ones(n) randn(n,10)*P]
    β = [randn(6); zeros(5)]
    σ = 1.0
    y = x*β + σ.*randn(n)
    return y, x, [β; σ]
end    

# the moments
function OLSmoments(θ::Array{Float64,1}, y::Array{Float64,1}, x::Array{Float64,2})
    n,k = size(x)
    β = θ[1:k]
    σ = θ[end]
    ϵhat = y - x*β # residuals
    ms = [x.*ϵhat ϵhat.^2 .- σ^2] # first k moments for β, last for σ 
    return ms
end

# prior: uniform
function prior(θ)
    lb = [-10.0*ones(11); 0.0]
    ub = [10.0*ones(11); 10.0]
    all((θ .>= lb) .& (θ .<= ub))
end

# this function generates a draw from the prior
function sample_from_prior()
	θ = rand(11)
    lb = [-10.0*ones(11); 0.0]
    ub = [10.0*ones(11); 10.0]
    theta = (ub-lb).*θ + lb
end


# single dimension random walk proposal
function proposal1(current, tuning)
    trial = copy(current)
    i = rand(1:size(current,1))
    trial[i] = current[i] + tuning[i].*randn()
    return trial
end

# MVN random walk proposal
function proposal2(current, cholV)
    current + cholV'*randn(size(current))
end

function MakeZ(θ, y, x, option)
    # make ghat
    ms = OLSmoments(θ, y, x)
    ghat = mean(ms,1)
    # compute ghats
    ghats = 1.0
    if option == "Gaussian"
        Σ = cov(ms)
        cholΣ = chol(Σ)
        ghats  = randn(size(ghat))*cholΣ/sqrt(n)
    end    
    if option == "Bootstrap"
        dx = size(x,2)
        data = [y x]
        data = bootstrap(data)
        y = data[:,1]
        x = data[:,2:end]
        mb = OLSmoments(θ, y, x)
        ghats = mean(mb,1) - ghat
    end    
    # final return
    Z = ghats - ghat
end

function logL(θ, y, x)
    n = size(y,1)
    # make ̂g(θ)
    ms = OLSmoments(θ, y, x)
    Econometrics.trim!(ms)
    ghat = mean(ms,1)
    # compute likelihood
    Σinv = inv(cov(ms))
    lnL = log(sqrt(det(Σinv))) - 0.5*n*(ghat*Σinv*ghat')[1]
    #lnL = - 0.5*n*(ghat*Σinv*ghat')[1]
end


