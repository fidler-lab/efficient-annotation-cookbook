// Empty JS for your own code to be here
$('#run-btn').on('click', function (e) {
    var n_workers = $('#select-n_workers').val();
    var n_gold_q = $('#select-n_gold_q').val();
    var agg = $('#select-agg').val();
    var features = $('#select-features').val();
    var model_freq = $('#select-model_freq').val();
    var use_ta = $('#checkbox-ta').is(":checked");
    var use_early_stop = $('#checkbox-early_stop').is(":checked");

    console.log(use_ta);
    console.log(use_early_stop);
    console.log(agg);
    console.log(n_gold_q);
    console.log(model_freq);
    console.log(features);


    if (n_workers.length == 0 || n_gold_q.length == 0 || agg.length == 0){
        msg = 'Some items are empty'
        window.alert(msg);
    }
    else if ((agg == 'Ours: w/ Semi-supervised CV model' || agg == 'Prior work: w/ CV model') && (features.length == 0 || model_freq.length == 0)) {
        msg = 'CV mode is used. Please select the features and the model update frequency'
        window.alert(msg);
    }
    else {
        // Change the src based on variable given
        var parent = $("embed#html-results").parent();
        var newElement = "<embed id='html-results' type='text/html' src='assets/results/example_plot.html' width=100% height=100%>";
        $("embed#html-results").remove();
        parent.append(newElement);    
    }
})