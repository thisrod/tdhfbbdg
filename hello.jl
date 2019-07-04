"""Sieve of Erastosthenes function docstring"""
function es(n::Int)
	isprime = trues(n)
	isprime[1] = false
	for i in 2:isqrt(n)
		if isprime[i]
			for j in i^2:i:n
				isprime[j] = false
			end
		end
	end
	return filter(x -> isprime[x], 1:n)
end