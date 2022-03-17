include("poisson2D.jl")
analytic_sol_func(x1,x2) = (sinpi(x1)*sinpi(x2))/(2pi^2)

## poisson equation in [0,1]^2
## Δu = -sin(πx1)sin(πx2)
## u = 0 at boundary

bc_func(x1, x2) = 0.0
rhs_func(x1, x2) = -sinpi(x1)*sinpi(x2)

# initial neural network
NN = paddle.nn.Sequential(
           paddle.nn.Linear(2, 16),
           paddle.nn.Sigmoid(),
           paddle.nn.Linear(16, 16),
           paddle.nn.Sigmoid(),
           paddle.nn.Linear(16, 1)
       )

# set batch size = 100
batch_size = 100
# sample 10 points from each side of Rectangle, so sample 40 point from boundary each iteration
bc_size = floor(Int, batch_size/10) 


# initial an optimizer with lr 0.1
adam = paddle.optimizer.Adam(learning_rate=0.1,
                    parameters=NN.parameters())

# tarining, 4000 iterations with lr 0.1
training(NN, adam, 4000, rhs_func, bc_func, batch_size, bc_size)

# initial an optimizer with lr 0.1
adam = paddle.optimizer.Adam(learning_rate=0.01,
                    parameters=NN.parameters())

# tarining, 2000 iterations with lr 0.01
training(NN, adam, 2000, rhs_func, bc_func, batch_size, bc_size)


# plot
x1s = Vector(0:0.01:1)
x2s = Vector(0:0.01:1)
u_real = analytic_sol_func.(x1s,x2s')

mesh(x1,x2) = [x1,x2]
inputs = mesh.(x1s, x2s')
inputs = reshape(inputs,:)
ins = zeros(101*101,2)
for i in 1:101*101
    ins[i,1] = inputs[i][1]
    ins[i,2] = inputs[i][2]
end
ins = paddle.to_tensor(ins)
u_predict = NN(ins).numpy()
u_predict = reshape(u_predict,101,101)
diff_u = abs.(u_predict .- u_real)

using LinearAlgebra
L2error = norm(u_predict .- u_real)

using Plots
p1 = plot(x1s, x2s, u_real, linetype=:contourf,title = "analytic");
p2 = plot(x1s, x2s, u_predict, linetype=:contourf,title = "predict");
p3 = plot(x1s, x2s, diff_u,linetype=:contourf,title = "error");
plot(p1,p2,p3)
savefig("poisson_paddle")
