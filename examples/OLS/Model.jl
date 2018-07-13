#using LinearAlgebra
using Econometrics

# dgp: CLRM with collinearity between regressors and additional instruments
# the optimal instruments are the regressors, but may be hard to determine  that
function dgp(n::Int64)
    # create covariances of regressors/instruments
    dimZ = 30 # number of instruments
    P = 2*rand(dimZ,dimZ) .- 1
    for i in 1:dimZ P[i,i] = 1.0 end
    P = UpperTriangular(P)
    # generate data and parameters
    z = [ones(n) randn(n,dimZ)*P]
    x = z[:,1:6]
    β = randn(6)
    σ = 1.0
    y = x*β + σ.*randn(n)
    return y, z, [β; σ]
end    

# the moments: interact errors with all instruments
function OLSmoments(θ::Array{Float64,1}, y::Array{Float64,1}, z::Array{Float64,2})
    k = size(θ,1)-1
    n = size(y,1)
    β = θ[1:k]
    σ = θ[end]
    x = z[:,1:k]
    ϵhat = y - x*β # residuals
    ms = [z.*ϵhat (n/(n-1)).*(ϵhat.^2 .- σ^2)] # first k moments for β, last for σ 
    return ms
end

# prior: uniform
function prior(θ)
    lb = [-10.0*ones(6); 0.0]
    ub = [10.0*ones(6); 10.0]
    all((θ .>= lb) .& (θ .<= ub))
end

# this function generates a draw from the prior
function sample_from_prior()
	θ = rand(11)
    lb = [-10.0*ones(6); 0.0]
    ub = [10.0*ones(6); 10.0]
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

function MakeZ(θ, y, z, option)
    # make ghat
    ms = OLSmoments(θ, y, z)
    ghat = mean(ms,1)
    # compute ghats
    ghats = 1.0
    if option == "Gaussian"
        Σ = cov(ms)
        cholΣ = chol(Σ)
        ghats  = randn(size(ghat))*cholΣ/sqrt(n)
    end    
    if option == "Bootstrap"
        data = [y z]
        data = bootstrap(data)
        y = data[:,1]
        z = data[:,2:end]
        mb = OLSmoments(θ, y, z)
        ghats = mean(mb,1) - ghat
    end    
    # final return
    Z = ghats - ghat
end

function logL(θ, y, z)
    n = size(y,1)
    # make ̂g(θ)
    ms = OLSmoments(θ, y, z)
    Econometrics.trim!(ms, 0.005)
    ghat = mean(ms,1)
    # compute likelihood
    Σinv = inv(cov(ms))
    lnL = log(sqrt(det(Σinv))) - 0.5*n*(ghat*Σinv*ghat')[1]
    #lnL = - 0.5*n*(ghat*Σinv*ghat')[1]
end

