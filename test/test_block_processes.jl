@testset "Block processes" begin
  m = 250
  n = 500
  k = 20
  p = 4
  s = 4
  verbose = false

  for FC in (Float64, ComplexF64)
    nbits_FC = sizeof(FC)
    nbits_I = sizeof(Int)

    @testset "Data Type: $FC" begin

      @testset "Block Hermitian Lanczos" begin
        A = rand(FC, n, n)
        A = A' * A
        B = rand(FC, n, p)

        @testset "algo = $algo" for algo in ("householder", "gs", "mgs", "givens")
          V, Ψ₁, T = hermitian_lanczos(A, B, k; algo)

          @test norm(V[:,1:p*s]' * V[:,1:p*s] - I) ≤ 1e-4
          @test V[:,1:p] * Ψ₁ ≈ B
          @test A * V[:,1:p*k] ≈ V * T
        end

        @testset "memory" begin
          function storage_block_hermitian_lanczos_bytes(n, p, k)
            nnzT = p*p + (k-1)*p*(2*p+1) + div(p*(p+1), 2)
            memory = 0
            memory += (p*k+1) * nbits_I       # Tₖ₊₁.ₖ -- colptr
            memory += nnzT * nbits_I          # Tₖ₊₁.ₖ -- rowval
            memory += nnzT * nbits_FC         # Tₖ₊₁.ₖ -- nzval
            memory += p*p * nbits_FC          # Ψ₁
            memory += p*p * nbits_FC          # Ωᵢ, Ψᵢ and Ψᵢ₊₁
            memory += n*p * (k+1) * nbits_FC  # Vₖ₊₁
            memory += p*n * nbits_FC          # q
            return memory
          end

          expected_block_hermitian_lanczos_bytes = storage_block_hermitian_lanczos_bytes(n, p, k)
          actual_block_hermitian_lanczos_bytes = @allocated hermitian_lanczos(A, B, k; algo="mgs")
          verbose && println("Block Hermitian Lanczos | $FC")
          verbose && println(expected_block_hermitian_lanczos_bytes, " ≤ ", actual_block_hermitian_lanczos_bytes, " ≤ ", 1.02 * expected_block_hermitian_lanczos_bytes, " ?")
          verbose && println()
          if VERSION < v"1.11.5" || !Sys.isapple()
            @test expected_block_hermitian_lanczos_bytes ≤ actual_block_hermitian_lanczos_bytes ≤ 1.02 * expected_block_hermitian_lanczos_bytes
          end
        end
      end

      @testset "Block Non-Hermitian Lanczos" begin
        A = rand(FC, n, n)
        B = rand(FC, n, p)
        C = rand(FC, n, p)
        V, Ψ₁, T, U, Φ₁ᴴ, Tᴴ = nonhermitian_lanczos(A, B, C, k)

        @test norm(V[:,1:p*s]' * U[:,1:p*s] - I) ≤ 1e-4
        @test norm(U[:,1:p*s]' * V[:,1:p*s] - I) ≤ 1e-4
        @test V[:,1:p] * Ψ₁ ≈ B
        @test U[:,1:p] * Φ₁ᴴ ≈ C
        @test T[1:k*p,1:k*p] ≈ Tᴴ[1:k*p,1:k*p]'
        @test A  * V[:,1:k*p] ≈ V * T
        @test A' * U[:,1:k*p] ≈ U * Tᴴ

        @testset "memory" begin
          # storage_block_nonhermitian_lanczos_bytes(n, p, k) = ...
          #
          #   expected_block_nonhermitian_lanczos_bytes = storage_block_nonhermitian_lanczos_bytes(n, p, k)
          #   actual_block_nonhermitian_lanczos_bytes = @allocated nonhermitian_lanczos(A, B, C, k)
          #   verbose && println("Block non-Hermitian Lanczos")
          #   verbose && println(expected_block_nonhermitian_lanczos_bytes, " ≤ ", actual_block_nonhermitian_lanczos_bytes, " ≤ ", 1.02 * expected_block_nonhermitian_lanczos_bytes, " ?")
          #   verbose && println()
          #   @test expected_block_nonhermitian_lanczos_bytes ≤ actual_block_nonhermitian_lanczos_bytes ≤ 1.02 * expected_block_nonhermitian_lanczos_bytes
        end
      end

      @testset "Block Arnoldi" begin
        A = rand(FC, n, n)
        B = rand(FC, n, p)

        @testset "algo = $algo" for algo in ("householder", "gs", "mgs", "givens")
          @testset "reorthogonalization = $reorthogonalization" for reorthogonalization in (false, true)
            V, Γ, H = arnoldi(A, B, k; algo, reorthogonalization)

            @test norm(V[:,1:p*s]' * V[:,1:p*s] - I) ≤ 1e-4
            @test V[:,1:p] * Γ ≈ B
            @test A * V[:,1:p*k] ≈ V * H
          end
        end

        @testset "memory -- reorthogonalization = $reorthogonalization" for reorthogonalization in (false, true)
          function storage_block_arnoldi_bytes(n, p, k)
            memory = 0
            memory += p*n * (k+1) * nbits_FC    # Vₖ₊₁
            memory += n*p * nbits_FC            # q
            memory += p*k * p*(k+1) * nbits_FC  # Hₖ₊₁.ₖ
            memory += p*p * nbits_FC            # Γ
            memory += p*p * nbits_FC            # Ψᵢⱼ and Ψtmp
            return memory
          end

          expected_block_arnoldi_bytes = storage_block_arnoldi_bytes(n, p, k)
          actual_block_arnoldi_bytes = @allocated arnoldi(A, B, k; algo="mgs", reorthogonalization)
          verbose && println("Block Arnoldi | $FC")
          verbose && println(expected_block_arnoldi_bytes, " ≤ ", actual_block_arnoldi_bytes, " ≤ ", 1.02 * expected_block_arnoldi_bytes, " ?")
          verbose && println()
          if VERSION < v"1.11.5" || !Sys.isapple()
            @test expected_block_arnoldi_bytes ≤ actual_block_arnoldi_bytes ≤ 1.02 * expected_block_arnoldi_bytes
          end
        end
      end

      @testset "Block Golub-Kahan" begin
        A = rand(FC, m, n)
        B = rand(FC, m, p)

        @testset "algo = $algo" for algo in ("householder", "gs", "mgs", "givens")
          V, U, Ψ₁, L = golub_kahan(A, B, k; algo)
          BL = L[1:(k+1)*p,1:k*p]

          @test norm(V[:,1:p*s]' * V[:,1:p*s] - I) ≤ 1e-4
          @test norm(U[:,1:p*s]' * U[:,1:p*s] - I) ≤ 1e-4
          @test U[:,1:p] * Ψ₁ ≈ B
          @test A  * V[:,1:k*p] ≈ U * BL
          @test A' * U ≈ V * L'
          @test A' * A  * V[:,1:k*p] ≈ V * L' * BL
          @test A  * A' * U[:,1:k*p] ≈ U * BL * L[1:k*p,1:k*p]'
        end

        @testset "memory" begin
          function storage_block_golub_kahan_bytes(m, n, p, k)
            nnzL = p*k*(p+1) + div(p*(p+1), 2)
            memory = 0
            memory += (p*(k+1)+1) * nbits_I       # Lₖ₊₁ -- colptr
            memory += nnzL * nbits_I              # Lₖ₊₁ -- rowval
            memory += nnzL * nbits_FC             # Lₖ₊₁ -- nzval
            memory += p*p * nbits_FC              # Ψ₁
            memory += p*p * nbits_FC              # Ψᵢ₊₁, TΩᵢ and TΩᵢ₊₁
            memory += p*(n+m) * (k+1) * nbits_FC  # Vₖ₊₁ and Uₖ₊₁
            memory += p*(n+m) * nbits_FC          # qᵥ and qᵤ
            return memory
          end

          expected_block_golub_kahan_bytes = storage_block_golub_kahan_bytes(m, n, p, k)
          actual_block_golub_kahan_bytes = @allocated golub_kahan(A, B, k; algo="mgs")
          verbose && println("Block Golub-Kahan | $FC")
          verbose && println(expected_block_golub_kahan_bytes, " ≤ ", actual_block_golub_kahan_bytes, " ≤ ", 1.02 * expected_block_golub_kahan_bytes, " ?")
          verbose && println()
          if VERSION < v"1.11.5" || !Sys.isapple()
            @test expected_block_golub_kahan_bytes ≤ actual_block_golub_kahan_bytes ≤ 1.02 * expected_block_golub_kahan_bytes
          end
        end
      end

      @testset "Block Saunders-Simon-Yip" begin
        A = rand(FC, m, n)
        B = rand(FC, m, p)
        C = rand(FC, n, p)

        @testset "algo = $algo" for algo in ("householder", "gs", "mgs", "givens")
          V, Ψ₁, T, U, Φ₁ᴴ, Tᴴ = saunders_simon_yip(A, B, C, k; algo)

          @test norm(V[:,1:p*s]' * V[:,1:p*s] - I) ≤ 1e-4
          @test norm(U[:,1:p*s]' * U[:,1:p*s] - I) ≤ 1e-4
          @test V[:,1:p] * Ψ₁ ≈ B
          @test U[:,1:p] * Φ₁ᴴ ≈ C
          @test T[1:k*p,1:k*p] ≈ Tᴴ[1:k*p,1:k*p]'
          @test A  * U[:,1:k*p] ≈ V * T
          @test A' * V[:,1:k*p] ≈ U * Tᴴ
          @test A' * A  * U[:,1:(k-1)*p] ≈ U * Tᴴ * T[1:k*p,1:(k-1)*p]
          @test A  * A' * V[:,1:(k-1)*p] ≈ V * T * Tᴴ[1:k*p,1:(k-1)*p]
        end

        @testset "memory" begin
          function storage_block_saunders_simon_yip_bytes(m, n, p, k)
            nnzT = p*p + (k-1)*p*(2*p+1) + div(p*(p+1), 2)
            memory = 0
            memory += (p*k+1) * nbits_I           # Tₖ₊₁.ₖ and (Tₖ.ₖ₊₁)ᴴ -- colptr
            memory += nnzT * nbits_I              # Tₖ₊₁.ₖ and (Tₖ.ₖ₊₁)ᴴ -- rowval
            memory += 2 * nnzT * nbits_FC         # Tₖ₊₁.ₖ and (Tₖ.ₖ₊₁)ᴴ -- nzval
            memory += 2 * p*p * nbits_FC          # Ψ₁ and Φ₁ᴴ
            memory += p*p * nbits_FC              # Ωᵢ, Ψᵢ, Ψᵢ₊₁, TΦᵢ and TΦᵢ₊₁
            memory += p*(n+m) * (k+1) * nbits_FC  # Vₖ₊₁ and Uₖ₊₁
            memory += p*(n+m) * nbits_FC          # qᵥ and qᵤ
          end

          expected_block_saunders_simon_yip_bytes = storage_block_saunders_simon_yip_bytes(m, n, p, k)
          actual_block_saunders_simon_yip_bytes = @allocated saunders_simon_yip(A, B, C, k; algo="mgs")
          verbose && println("Block Saunders-Simon-Yip")
          verbose && println(expected_block_saunders_simon_yip_bytes, " ≤ ", actual_block_saunders_simon_yip_bytes, " ≤ ", 1.02 * expected_block_saunders_simon_yip_bytes, " ?")
          verbose && println()
          if VERSION < v"1.11.5" || !Sys.isapple()
            @test expected_block_saunders_simon_yip_bytes ≤ actual_block_saunders_simon_yip_bytes ≤ 1.02 * expected_block_saunders_simon_yip_bytes
          end
        end
      end

      @testset "Block Montoison-Orban" begin
        A = rand(FC, m, n)
        B = rand(FC, n, m)
        D = rand(FC, m, p)
        C = rand(FC, n, p)

        @testset "algo = $algo" for algo in ("householder", "gs", "mgs", "givens")
          @testset "reorthogonalization = $reorthogonalization" for reorthogonalization in (false, true)
            V, Γ, H, U, Λ, F = montoison_orban(A, B, D, C, k; algo, reorthogonalization)

            @test norm(V[:,1:p*s]' * V[:,1:p*s] - I) ≤ 1e-4
            @test norm(U[:,1:p*s]' * U[:,1:p*s] - I) ≤ 1e-4
            @test V[:,1:p] * Γ ≈ D
            @test U[:,1:p] * Λ ≈ C
            @test A * U[:,1:k*p] ≈ V * H
            @test B * V[:,1:k*p] ≈ U * F
            @test B * A * U[:,1:(k-1)*p] ≈ U * F * H[1:k*p,1:(k-1)*p]
            @test A * B * V[:,1:(k-1)*p] ≈ V * H * F[1:k*p,1:(k-1)*p]
          end
        end

        @testset "memory -- reorthogonalization = $reorthogonalization" for reorthogonalization in (false, true)
          function storage_block_montoison_orban_bytes(m, n, p, k)
            memory = 0
            memory += 2 * p*p * nbits_FC            # Γ and Λ
            memory += p*p * nbits_FC                # Ψᵢⱼ, Φᵢⱼ, Ψtmp and Φtmp
            memory += 2 * p*k * p*(k+1) * nbits_FC  # Hₖ₊₁.ₖ and Fₖ₊₁.ₖ
            memory += p*(n+m) * (k+1) * nbits_FC    # Vₖ₊₁ and Uₖ₊₁
            memory += p*(n+m)* nbits_FC             # qᵥ and qᵤ
            return memory
          end

          expected_block_montoison_orban_bytes = storage_block_montoison_orban_bytes(m, n, p, k)
          actual_block_montoison_orban_bytes = @allocated montoison_orban(A, B, D, C, k; algo="mgs", reorthogonalization)
          verbose && println("Block Montoison-Orban | $FC")
          verbose && println(expected_block_montoison_orban_bytes, " ≤ ", actual_block_montoison_orban_bytes, " ≤ ", 1.02 * expected_block_montoison_orban_bytes, " ?")
          verbose && println()
          if VERSION < v"1.11.5" || !Sys.isapple()
            @test expected_block_montoison_orban_bytes ≤ actual_block_montoison_orban_bytes ≤ 1.02 * expected_block_montoison_orban_bytes
          end
        end
      end
    end
  end
end
