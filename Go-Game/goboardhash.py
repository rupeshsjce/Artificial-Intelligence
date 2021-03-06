from gohelper import Player, Point

__all__ = ['HASH_CODE', 'EMPTY_BOARD']

HASH_CODE = {
    (Point(row=1, col=1), None): 1960111529681953302,
    (Point(row=1, col=1), Player.black): 7764354851860950237,
    (Point(row=1, col=1), Player.white): 568346988098392042,
    (Point(row=1, col=2), None): 6087972728423602540,
    (Point(row=1, col=2), Player.black): 116923585788596382,
    (Point(row=1, col=2), Player.white): 8030915595376234706,
    (Point(row=1, col=3), None): 2278105690188580342,
    (Point(row=1, col=3), Player.black): 350039180611194257,
    (Point(row=1, col=3), Player.white): 2229576233707756074,
    (Point(row=1, col=4), None): 7619426613501116179,
    (Point(row=1, col=4), Player.black): 4761610172306478649,
    (Point(row=1, col=4), Player.white): 4215635203768342861,
    (Point(row=1, col=5), None): 1992491069829767579,
    (Point(row=1, col=5), Player.black): 7006904175735913593,
    (Point(row=1, col=5), Player.white): 1382554074055987460,
    (Point(row=2, col=1), None): 8932790082662620608,
    (Point(row=2, col=1), Player.black): 3770856473684386478,
    (Point(row=2, col=1), Player.white): 3101822738308334582,
    (Point(row=2, col=2), None): 7102686148743152097,
    (Point(row=2, col=2), Player.black): 5459022478148370694,
    (Point(row=2, col=2), Player.white): 1330005438117062937,
    (Point(row=2, col=3), None): 6483783885815010207,
    (Point(row=2, col=3), Player.black): 575083460855124836,
    (Point(row=2, col=3), Player.white): 8908010048329696597,
    (Point(row=2, col=4), None): 4014034866107802936,
    (Point(row=2, col=4), Player.black): 4634747692305341751,
    (Point(row=2, col=4), Player.white): 5099962446257730365,
    (Point(row=2, col=5), None): 33090496880301691,
    (Point(row=2, col=5), Player.black): 4078036160807883613,
    (Point(row=2, col=5), Player.white): 63100367930907885,
    (Point(row=3, col=1), None): 5424286213673281038,
    (Point(row=3, col=1), Player.black): 7237011017940221426,
    (Point(row=3, col=1), Player.white): 5028966271191621651,
    (Point(row=3, col=2), None): 1920783851670008211,
    (Point(row=3, col=2), Player.black): 1447831806447066175,
    (Point(row=3, col=2), Player.white): 9006386422033921678,
    (Point(row=3, col=3), None): 2350473377366737857,
    (Point(row=3, col=3), Player.black): 7726692912363248905,
    (Point(row=3, col=3), Player.white): 1823736794501952709,
    (Point(row=3, col=4), None): 6909028843380999294,
    (Point(row=3, col=4), Player.black): 6737039599794784095,
    (Point(row=3, col=4), Player.white): 1724091370170135532,
    (Point(row=3, col=5), None): 3427160494306447097,
    (Point(row=3, col=5), Player.black): 4839267201970345761,
    (Point(row=3, col=5), Player.white): 1821289944111512541,
    (Point(row=4, col=1), None): 936425366560978800,
    (Point(row=4, col=1), Player.black): 8474301863347863528,
    (Point(row=4, col=1), Player.white): 9050324195416421344,
    (Point(row=4, col=2), None): 5424105521634150665,
    (Point(row=4, col=2), Player.black): 3141022339753875167,
    (Point(row=4, col=2), Player.white): 8138042960241750988,
    (Point(row=4, col=3), None): 1734801857829946570,
    (Point(row=4, col=3), Player.black): 1991230347982436319,
    (Point(row=4, col=3), Player.white): 6476625625492580589,
    (Point(row=4, col=4), None): 5989553919477028527,
    (Point(row=4, col=4), Player.black): 585178816610768471,
    (Point(row=4, col=4), Player.white): 4059036797343930912,
    (Point(row=4, col=5), None): 2328902826039655929,
    (Point(row=4, col=5), Player.black): 5098545479306880604,
    (Point(row=4, col=5), Player.white): 3163028099314853781,
    (Point(row=5, col=1), None): 6027530747759182395,
    (Point(row=5, col=1), Player.black): 485603356742570164,
    (Point(row=5, col=1), Player.white): 4787372277887945666,
    (Point(row=5, col=2), None): 2081566055906421455,
    (Point(row=5, col=2), Player.black): 5506791157410929249,
    (Point(row=5, col=2), Player.white): 2414114900353311398,
    (Point(row=5, col=3), None): 80052661181608161,
    (Point(row=5, col=3), Player.black): 4396659849194325268,
    (Point(row=5, col=3), Player.white): 6424558200131379814,
    (Point(row=5, col=4), None): 8257228903786131256,
    (Point(row=5, col=4), Player.black): 1001523182804354481,
    (Point(row=5, col=4), Player.white): 6891071925603793339,
    (Point(row=5, col=5), None): 6538776104684239439,
    (Point(row=5, col=5), Player.black): 5955499212060837645,
    (Point(row=5, col=5), Player.white): 6363772611000398190,
}

EMPTY_BOARD = 9181944435492932547
