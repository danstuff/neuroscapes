Neuron::Neuron(uint num_ipts, uint thresh){
    num_inputs = num_ipts;
    threshold = thresh;
    
    assert(num_inputs <= NEU_MAX_CON); 

    clearInputs();
}

void Neuron::clearInputs(){
    for(uint i = 0; i < num_inputs; i++){
        inputs[i] = false;
    }
}

void Neuron::setInput(uint i){
   assert(i < num_inputs); 
   inputs[i] = true;
}

void Neuron::unsetInput(uint i){
   assert(i < num_inputs); 
   inputs[i] = false;
}

bool Neuron::getOutput(){
    //get a sum of all enabled inputs
    uint num_enabled = 0;

    for(uint i = 0; i < num_inputs; i++){
        if(inputs[i]){ num_enabled++; }
    }
    
    //if number of enabled inputs is >= threshold, return true
    return (num_enabled >= threshold);
}
