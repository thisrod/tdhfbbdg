# Berry phase routines

using LinearAlgebra: dot

"pci([q1, q2, ...]) pointwise Berry phase after sequence of states"
function pci(S,T)
    @assert length(S) == length(T)
    if length(S) == 1
        zero(S[])
    else
        [@. imag(conj(S[j+1])*T[j]) for j = 1:length(S)-1] |> sum
    end
end
pci(S) = pci(S, S)

"bphase([q1, q2, ...]) cumulative Berry phase for sequence of states"
function bphase(S,T)
    bp = [dot(S[j+1], T[j]) |> imag for j = 1:length(S)-1] |> cumsum
    [0; bp]
end
bphase(S) = bphase(S,S)
