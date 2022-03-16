using PyCall
paddle = pyimport("paddle")
batch_hessian = paddle.autograd.batch_hessian
mes_loss = paddle.nn.MSELoss()
paddle.set_default_dtype("float64")

function sample_domain(batch_size::Int)
    x = rand(batch_size, 2)
    return x
end

function sample_bc(batch_size::Int)
    zero_v = zeros(batch_size)
    one_v = ones(batch_size)
    ins1 = [zero_v rand(batch_size)]
    ins2 = [rand(batch_size) zero_v]
    ins3 = [one_v rand(batch_size)]
    ins4 = [rand(batch_size) one_v]
    x = [ins1;ins2;ins3;ins4]
    return x
end

function sample_to_tensor(rhs_func::Function, bc_func::Function, batch_size::Int, bc_size::Int)
    # sample uniformly from domain with size (batch_size, 2)
    x = sample_domain(batch_size)
    rhs = rhs_func.(x[:,1], x[:,2])
    rhs = reshape(rhs,batch_size,1)

    # sample uniformly from boundary with size (bc_size, 2)
    bc_x = sample_bc(bc_size)
    bc_value = bc_func.(bc_x[:,1],bc_x[:,2])
    bc_value = reshape(bc_value,4*bc_size,1)

    x = paddle.to_tensor(x)
    rhs = paddle.to_tensor(rhs)
    bc_x = paddle.to_tensor(bc_x)
    bc_value = paddle.to_tensor(bc_value)
    return (x, rhs, bc_x, bc_value)
end

# Is there a better way to get the slice of tensor with PyCall? 
py"""
def get_slice(x,i):
    return x[i,:,i]
"""

function loss_func(NN, x, rhs, bc_x, bc_value)
    batch_size = x.shape[1]
    x.stop_gradient = false

    # using the auto diff, also could use finite difference instead
    d2u_dx2 = batch_hessian(NN, [x], create_graph=true)
    d2u_dx2 = paddle.reshape(d2u_dx2, shape=PyVector([2, batch_size, 2]))
    Laplace = py"get_slice"(d2u_dx2,0) + py"get_slice"(d2u_dx2,1)
    Laplace = paddle.reshape(Laplace, shape=PyVector([batch_size, 1]))

    # Poisson Equation Δu = f(x)
    loss = mes_loss(Laplace,rhs)
    x.stop_gradient = true

    # boundary condition: u(x) = g(x)
    NN_value = NN(bc_x)
    loss += mes_loss(NN_value, bc_value)
    
    return loss
end

function training(NN, opt, iterations::Int, rhs_func::Function, bc_func::Function, batch_size::Int, bc_size::Int)
    for iter in 1:iterations
        x, rhs, bc_x, bc_value = sample_to_tensor(rhs_func, bc_func, batch_size, bc_size)
        loss = loss_func(NN, x, rhs, bc_x, bc_value)
        println(loss.numpy()[1])
        loss.backward()
        opt.step()
        opt.clear_grad()
    end
end
