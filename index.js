function doThing() {
    var theThing = 3;
    function doInnerThing(theThing) {
        var theSumator = 0;
        while (theThing--) {
            theSumator++;
        }
        return theSumator;
    }
    doInnerThing(theThing);
}

doThing();
