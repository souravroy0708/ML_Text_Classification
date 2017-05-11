//********************************* STARTs HERE ***********
//For User input form validation error message
var message;
function commonPnotify(type,message) {
    var opts = {
        shadow: false
    };
    switch (type) {
        case 'userInputValidationError':
            opts.title ="Error :)";
            opts.text = "Keyword cannot be blank";
            opts.type = "info";
            break;
        case 'newSearchValue':
            opts.title ="Success :)";
            opts.text = "New ULR has set successfully.Click on submit button to search again.";
            opts.type = "info";
            break;
        
     }
     new PNotify(opts);
  }


var form = $("#searchForm");
form.validate({
       rules: {
            keyword: {
              required:true,
            },
      },
        messages: {
            keyword:{
            required:"Keyword cannot be blank",
          },
          }
});



$(document).ready(function(){

  
    $('[data-toggle="tooltip"]').tooltip();
    $("#user_input_submit").click(function(){

      $(".record_tr").hide();
      var keyword = $("#keyword").val();
      var recordCount = $("#recordCount").val();
      // console.log("type"+type+"min_score"+min_score);
      $("#aue_not_found").hide();
      $("#aueTable").hide();
      formValidation=form.valid();
    
    if(formValidation == false){
        commonPnotify('userInputValidationError');
        }
       else{
  
  $("#aueLoadingSpinner").show();
  $(".loading_div").show();
  
    $.ajax({
    url : "/get-search-result/", 
    type : "GET", 
    timeout: '6000000000',
    data : { recordCount : recordCount,keyword : keyword}, 
    success : function(data) {
      $(".loading_div").hide();
      $("#aueLoadingSpinner").hide();
      var search_result = data["result"]
      console.log(search_result);
      counter  = 1
      for (i = 0; i < search_result.length; i++) {
        $("#aueTable").show();
        $("#aue_not_found").hide();
        $("#record_tr_"+counter).show();
        result_url = search_result[i]["url"]
        result_description =   search_result[i]["description"]
          $("#s_no_"+counter).text(counter);
          $("#url_no_"+counter).text(result_description);
          $("#action_no_"+counter).attr("href",result_url);
          // $("#set_search_with_"+counter).attr("set_search_value",result_url);
          counter ++;
        }
        if(search_result.length <= 0){
          $("#aue_not_found").show();
          $("#aueTable").hide();
        }
       
        },
    error : function(xhr,errmsg,err) {
      alert("Timeout error.Please check your internet connection or enter a valid url and then try again "+errmsg);
         console.log(xhr.status + ": " + xhr.responseText+"xhr"+xhr+"err"+err);
         
         $("#aueLoadingSpinner").hide();
         $(".loading_div").hide();
    }
});
}

});
});



