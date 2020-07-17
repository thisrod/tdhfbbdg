using Test

C = NaN
N = 10
l = rand()

include("system.jl")

crand(dims...) = randn(dims...) + 1im*(randn(dims...))

@testset "Wirtinger derivatives" begin
    v = crand(N,N)
    w = crand(N,N)
    a = crand()
    b = crand()
    Pv, Qv = poles(v)
    Pw, Qw = poles(w)
    Ps, Qs = poles(a*v+b*w)
    @test Ps ≈ a*Pv + b*Pw
    @test Qs ≈ a*Qv + b*Qw
    
    inones = zero(z)
    inones[2:end-1,2:end-1] .= 1
    P, Q = poles(ones(N,N))
    @test P ≈ zero(z) atol=1e-10
    @test Q ≈ zero(z) atol=1e-10
    P, Q = poles(z)
    @test P ≈ inones
    @test Q ≈ zero(z) atol=1e-10
    P, Q = poles(conj.(z))
    @test P ≈ zero(z) atol=1e-10
    @test Q ≈ inones    
end